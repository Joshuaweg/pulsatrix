/** @file uncertainty.hpp
 *  @brief How sure a model is of each prediction (KS-3): Monte Carlo dropout (Gal and
 *         Ghahramani, ICML 2016), deep ensembles (Lakshminarayanan, Pritzel and Blundell, NeurIPS
 *         2017), and the split of a prediction's uncertainty into the part more data would remove
 *         and the part it wouldn't.
 *  @ingroup interpretability
 *
 * One deterministic network gives one probability per class, and nothing says how much to trust
 * it. Several predictions of the same input that disagree do. Monte Carlo dropout gets them from one
 * network by keeping dropout on at inference; a deep ensemble from several networks trained from
 * different seeds. Either way the result is a set of samples of the predictive distribution, and
 * DecomposeUncertainty() splits its entropy:
 *
 *  - **total**, the entropy of the mean prediction, H[E p];
 *  - **aleatoric**, the mean entropy of the samples, E H[p]: what each one is unsure of, noise in
 *    the data that more data wouldn't remove;
 *  - **epistemic**, the difference, the mutual information between the prediction and the model
 *    (BALD; Houlsby et al., 2011): how much the samples disagree, which more data would reduce.
 *
 * For regression, each member predicts a mean and a variance, trained with GaussianNLLLoss, and
 * CombineGaussians() mixes them the same way.
 *
 * The decomposition is only as good as the samples. Dropout samples come from one network and
 * tend to agree more than an ensemble's; neither is the true posterior.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "pulsatrix/device_backend.hpp"
#include "pulsatrix/json.hpp"
#include "pulsatrix/module.hpp"
#include "pulsatrix/tensor.hpp"

namespace pulsatrix {

/**
 * @brief Monte Carlo dropout: runs @p predict on @p input @p samples times with every dropout
 *        layer under @p model in training mode and everything else in eval mode, so BatchNorm uses
 *        its running statistics and only the dropout masks change between passes.
 *
 * Dropout layers are the DropoutModule instances in the model's tree (NamedModules()) and the
 * LinearModule layers whose LoRA adapter has dropout (LoRAAdapter::dropout() > 0). Each module's training mode is restored
 * afterwards, also when predict throws. The masks come from each layer's seeded stream, so a
 * run is reproducible.
 * @param model The network @p predict runs.
 * @param predict Runs the network on its argument and returns probabilities, such as a softmax of
 *        the logits.
 * @return One output per pass.
 * @throws std::invalid_argument if samples < 2 or the model has no dropout layer with p > 0, whose
 *         samples would all be the same.
 */
[[nodiscard]] std::vector<Tensor> McDropoutSamples(Module& model, const std::function<Tensor(const Tensor&)>& predict,
                                                   const Tensor& input, int64_t samples);

/**
 * @brief A deep ensemble's samples: each member's output for @p input.
 * @param members Each runs one network, in eval mode, and returns probabilities. Train them from
 *        different seeds on the same data; Lakshminarayanan et al. used 5.
 * @throws std::invalid_argument if there are fewer than 2 members.
 */
[[nodiscard]] std::vector<Tensor> EnsembleSamples(const std::vector<std::function<Tensor(const Tensor&)>>& members,
                                                  const Tensor& input);

/** @brief DecomposeUncertainty()'s result: one value per input, entropies in nats. */
struct UncertaintyDecomposition {
    /** @brief The mean of the samples' probabilities, (N, K): the prediction to use. */
    Tensor mean;
    /** @brief The most probable class of the mean, the lower index on a tie. */
    std::vector<int64_t> predicted;
    /** @brief H[E p], the entropy of the mean. */
    std::vector<double> total;
    /** @brief E H[p], the samples' mean entropy. */
    std::vector<double> aleatoric;
    /** @brief total - aleatoric, the mutual information (never negative). */
    std::vector<double> epistemic;
    /** @brief The fraction of samples whose most probable class is the mean's. */
    std::vector<double> agreement;
    int64_t samples = 0;
};

/**
 * @brief Splits the uncertainty of a set of probability samples into its aleatoric and epistemic
 *        parts, input by input.
 * @param samples Two or more (N, K) probability tensors of the same shape, such as
 *        McDropoutSamples() or EnsembleSamples() returns.
 * @throws std::invalid_argument if there are fewer than 2 samples, they aren't all (N, K) of the
 *         same shape with K >= 2, or a probability is negative or not finite.
 */
[[nodiscard]] UncertaintyDecomposition DecomposeUncertainty(const std::vector<Tensor>& samples);

/** @brief CombineGaussians()' result: one value per input. */
struct GaussianMixturePrediction {
    /** @brief The mixture's mean, the members' mean. */
    std::vector<double> mean;
    /** @brief The mixture's variance: aleatoric + epistemic. */
    std::vector<double> variance;
    /** @brief The members' mean variance: the noise each predicts. */
    std::vector<double> aleatoric;
    /** @brief The variance of the members' means: how much they disagree. */
    std::vector<double> epistemic;
};

/**
 * @brief Mixes M Gaussian predictions per input into one, as Lakshminarayanan et al. do: the
 *        mixture's mean is the mean of the means, and its variance the mean of
 *        (variance + mean^2) minus the mixture mean squared.
 * @param means means[m][i]: member m's mean for input i.
 * @param variances The same shape, or empty for members that predict only a mean (Monte Carlo
 *        dropout of a plain regressor); the aleatoric part is then 0.
 * @throws std::invalid_argument if there are fewer than 2 members, the members have different
 *         lengths or none, a value isn't finite, or a variance is negative.
 */
[[nodiscard]] GaussianMixturePrediction CombineGaussians(const std::vector<std::vector<double>>& means,
                                                         const std::vector<std::vector<double>>& variances = {});

/**
 * @brief The Gaussian negative log likelihood, for regressors that predict their own noise
 *        (Nix and Weigend, 1994; Lakshminarayanan et al., 2017): the network outputs two values per
 *        input, a mean mu and s, and its variance is softplus(s) + min_variance, which keeps it
 *        positive.
 *
 * loss = mean_n 0.5 (log var_n + (y_n - mu_n)^2 / var_n), PyTorch's GaussianNLLLoss with
 * full=False on that variance.
 * @note Not a Module, like the other losses. It computes on the host and returns the gradient on
 *       the prediction's device.
 */
class GaussianNLLLoss {
public:
    /**
     * @param backend Backend to allocate the gradient through. Not owned; must outlive this loss.
     * @param min_variance Added to softplus(s); Lakshminarayanan et al. used 1e-6.
     * @throws std::invalid_argument if min_variance isn't positive.
     */
    explicit GaussianNLLLoss(DeviceBackend* backend, double min_variance = 1e-6);

    /**
     * @param prediction (N, 2): the mean, then s, for each input.
     * @param target (N, 1) or (N).
     * @return The loss.
     * @throws std::invalid_argument on other shapes or a value that isn't finite.
     */
    [[nodiscard]] float forward(const Tensor& prediction, const Tensor& target);

    /**
     * @brief The gradient with respect to the prediction, (N, 2).
     * @throws std::logic_error if forward() hasn't been called.
     */
    [[nodiscard]] Tensor backward() const;

    /** @brief The variance a prediction's s gives: softplus(s) + min_variance. */
    [[nodiscard]] double variance(double s) const;

private:
    DeviceBackend* backend_;
    double min_variance_;
    Tensor last_prediction_;
    std::vector<float> last_target_;
    bool has_forwarded_ = false;
};

/** @brief The decomposition as a `pulsatrix.uncertainty.v1` document, without the mean
 *         probabilities. */
[[nodiscard]] JsonValue ToJson(const UncertaintyDecomposition& decomposition);

}  // namespace pulsatrix
