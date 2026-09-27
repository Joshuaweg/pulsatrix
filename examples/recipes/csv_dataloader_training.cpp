/** @file csv_dataloader_training.cpp
 *  @brief Recipe: trains a small LinearModule regressor by pulling shuffled batches from a
 *         CSV file through CsvDataset + DataLoader, instead of hand-building Tensors in
 *         code (contrast with xor_training.cpp's hardcoded four-example dataset).
 *         Paired with docs/recipes/data-pipeline/csv_dataloader_training.md.
 */
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "pulsatrix/adam_optimizer.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/csv_dataset.hpp"
#include "pulsatrix/data_loader.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/mse_loss.hpp"

int main() {
    using namespace pulsatrix;

    CPUBackend backend;

    // A tiny synthetic regression dataset (y = 2*x1 - 3*x2 + 1, plus a touch of per-row
    // variation) written to a real CSV file -- recipes are self-contained, so this writes
    // its own fixture rather than depending on a path outside the recipe.
    const std::string csv_path = "csv_dataloader_training_recipe_data.csv";
    {
        std::ofstream out(csv_path);
        out << "x1,x2,y\n";
        for (int i = 0; i < 20; ++i) {
            float x1 = static_cast<float>(i) * 0.5f;
            float x2 = static_cast<float>(i % 5);
            float y = 2.0f * x1 - 3.0f * x2 + 1.0f;
            out << x1 << "," << x2 << "," << y << "\n";
        }
    }

    auto dataset =
        std::make_shared<CsvDataset>(csv_path, std::vector<std::string>{"x1", "x2"}, "y", &backend);

    DataLoaderOptions options;
    options.batch_size = 4;
    options.shuffle = true;
    options.shuffle_seed = 42;

    LinearModule model(/*in_features=*/2, /*out_features=*/1, &backend);
    AdamOptimizer optimizer(0.05f, &backend);
    MSELoss loss_fn(&backend);

    std::printf("CSV + DataLoader training recipe -- Linear(2,1) regressing y = 2*x1 - 3*x2 + 1\n");
    std::printf("%lld rows, batch_size=%lld, shuffled each epoch\n\n", static_cast<long long>(dataset->size()),
                static_cast<long long>(options.batch_size));

    constexpr int kEpochs = 50;
    for (int epoch = 1; epoch <= kEpochs; ++epoch) {
        DataLoader loader(dataset, &backend, options);  // fresh Sampler -> reshuffled order each epoch
        float total_loss = 0.0f;
        int64_t batches = 0;

        while (auto batch = loader.next_batch()) {
            optimizer.zero_grad(model);

            Tensor prediction = model.forward(batch->fields[0]);
            float loss_value = loss_fn.forward(prediction, batch->fields[1]);
            Tensor grad_prediction = loss_fn.backward();
            (void)model.backward(grad_prediction);

            optimizer.step(model);

            total_loss += loss_value;
            ++batches;
        }

        if (epoch == 1 || epoch % 10 == 0) {
            std::printf("epoch %2d | mean batch MSE %.4f\n", epoch, total_loss / static_cast<float>(batches));
        }
    }

    std::remove(csv_path.c_str());
    return 0;
}
