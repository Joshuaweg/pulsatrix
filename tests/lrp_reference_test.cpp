// LRP-completeness campaign Mission 3: whole-model LRP::explain() checked against the reference
// LRP implementations -- Zennit 1.0.0 (Part A: Epsilon / ZPlus / AlphaBeta / Gamma / ZBox and the
// EpsilonPlus / EpsilonAlpha2Beta1 / EpsilonGammaBox composites) and LXT 2.1's AttnLRP (Part B:
// MultiHeadAttentionModule and TransformerBlock), not just this project's own derivations.
// Reference values generated offline by tools/generate_lrp_reference_values.py inside the image
// built from tools/lrp_reference.Dockerfile (Python 3.12, torch 2.14.1+cpu, torchvision
// 0.29.1+cpu, numpy 2.5.2, zennit 1.0.0, lxt 2.1):
//   docker run --rm -v "$PWD":/w -w /w pulsatrix-lrpref:latest python3 tools/generate_lrp_reference_values.py
// and pasted below verbatim -- zero live Python dependency at test/build time.
//
// Shared parameters: every weight, bias and input is Det(n, mul, add, div) -- the generator's
// det(): float(((i * mul + add) % 17) - 8) / float(div), one IEEE float32 division on each side,
// so both sides run bit-identical networks. Linear weights are generated in pulsatrix's
// (in_features, out_features) order and transposed into torch.nn.Linear by the generator;
// Conv2D kernels are (out, in, kh, kw) on both sides (stride 1, no padding).
//
// Seeding: LRPSeed::OneHot (unit relevance at one target per row, a different target per row),
// which is what a Zennit / LXT attributor computes when handed a one-hot output gradient.
//
// Tolerance: float32 vs float32, |actual - expected| <= 1e-4 * max(1, |expected|). Observed
// max abs error when this was written: 1.2e-7 (MLP), 1.9e-6 (CNN), 1.3e-6 (attention), 8.7e-6
// (TransformerBlock, logits up to 25); the slack only absorbs float32 summation-order
// differences, never a convention difference.
//
// Conventions that must be matched to compare like with like (none is hidden by tolerance):
//  * Zennit's Epsilon puts the bias in the denominator, so the Epsilon cases use
//    LRPRuleConfig::epsilon_bias_in_denominator = true. pulsatrix's default epsilon rule
//    (bias left out of z) is a different, conservative rule and does NOT match Zennit when
//    biases are nonzero.
//  * Zennit's EpsilonGammaBox maps ZBox onto the first *convolution* only (its first_map holds
//    Convolution alone), so on a Conv-free MLP it is Epsilon everywhere. pulsatrix's preset
//    follows that (fixed in this mission; it used to put ZBox on a leading Linear).
//  * LXT's stabilizer is z + eps for every sign of z; pulsatrix's is z + eps * sign(z). With
//    eps = 1e-6 and no near-zero denominators in these models the two agree to float precision;
//    they would differ visibly only for |z| ~ eps.
//  * LXT's linear_epsilon keeps the bias in z, so Part B also uses epsilon_bias_in_denominator.
//    With the default pre-bias rule the attention projections give different (bias-free-z)
//    relevance; MultiHeadAttentionModule and TransformerBlock forward the flag to every inner
//    LinearModule, so the attention-specific rules (softmax Prop. 3.1, the matmul epsilon +
//    uniform split, RMSNorm identity, residual epsilon split, SwiGLU uniform split + SiLU
//    identity) are what Part B actually compares.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "pulsatrix/conv2d_module.hpp"
#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/explainer_context.hpp"
#include "pulsatrix/flatten_module.hpp"
#include "pulsatrix/linear_module.hpp"
#include "pulsatrix/lrp.hpp"
#include "pulsatrix/multihead_attention_module.hpp"
#include "pulsatrix/relu_module.hpp"
#include "pulsatrix/transformer_block.hpp"

namespace pulsatrix {
namespace {

// The generator's det(): float(((i * mul + add) % 17) - 8) / float(div).
std::vector<float> Det(int64_t n, int64_t mul, int64_t add, int64_t div) {
    std::vector<float> v(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        v[static_cast<size_t>(i)] = static_cast<float>((i * mul + add) % 17 - 8) / static_cast<float>(div);
    }
    return v;
}

// Every bias in the generator is det(out, 3, add, div).
void SetLinear(LinearModule& m, int64_t in, int64_t out, int64_t mul, int64_t add, int64_t div, int64_t bias_add,
               int64_t bias_div) {
    m.set_weight(Det(in * out, mul, add, div));
    m.set_bias(Det(out, 3, bias_add, bias_div));
}

void SetConv(Conv2DModule& m, int64_t in, int64_t out, int64_t k, int64_t mul, int64_t add, int64_t div,
             int64_t bias_add, int64_t bias_div) {
    m.set_kernel(Det(out * in * k * k, mul, add, div));
    m.set_bias(Det(out, 3, bias_add, bias_div));
}

void ExpectMatchesReference(const Tensor& actual, const std::vector<float>& expected) {
    ASSERT_EQ(static_cast<size_t>(actual.numel()), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        const float tol = 1e-4f * std::max(1.0f, std::fabs(expected[i]));
        EXPECT_NEAR(actual.data()[i], expected[i], tol) << "element " << i;
    }
}

LRPRuleConfig ZennitEpsilon(float eps) {
    LRPRuleConfig c{eps};
    c.epsilon_bias_in_denominator = true;
    return c;
}
LRPRuleConfig AlphaBetaRule(float alpha, float beta) {
    LRPRuleConfig c{1e-6f};
    c.rule = LRPRule::AlphaBeta;
    c.alpha = alpha;
    c.beta = beta;
    return c;
}
LRPRuleConfig GammaRule(float gamma) {
    LRPRuleConfig c{1e-6f};
    c.rule = LRPRule::Gamma;
    c.gamma = gamma;
    return c;
}

// ---- Reference values: tools/generate_lrp_reference_values.py output, verbatim ----
// ---- Part A: Zennit 1.0.0 ----
const std::vector<float> kMlpEpsilon1e6 = {
    -0.415450871f, 0.0f, 0.166922331f, 0.726000845f, 0.0578664057f, 0.288887084f, 0.264229447f, 0.177596867f,
    -0.00216581812f, 0.129948959f, 0.224161908f, -0.00487308204f
};
const std::vector<float> kMlpEpsilon025 = {
    -0.231960744f, 0.0f, 0.120200112f, 0.499264747f, 0.0386141054f, 0.22423251f, 0.200627849f, 0.137323484f,
    -0.00192576961f, 0.101998858f, 0.173580125f, -0.00347508048f
};
const std::vector<float> kMlpZPlus = {
    0.165092289f, 0.0f, 0.184430927f, 0.25503394f, 0.00448016962f, 0.104827963f, 0.237484381f, 0.156082869f,
    0.00509816548f, 0.156904697f, 0.196648166f, 0.00332136871f
};
const std::vector<float> kMlpAlphaBeta21 = {
    -0.8662889f, 0.0f, -0.117050022f, 1.60409713f, 0.188600004f, 1.88940895f, 0.820607305f, 0.757887244f,
    -1.28963685f, 0.434346437f, 0.977532089f, -0.00695603341f
};
const std::vector<float> kMlpGamma025 = {
    -0.0697718933f, 0.0f, 0.173353821f, 0.488822311f, 0.0253746547f, 0.199405834f, 0.264463484f, 0.173968777f,
    -0.000251332502f, 0.126904488f, 0.219198555f, -0.00343409623f
};
const std::vector<float> kMlpEpsilonPlus = {
    -0.415450871f, 0.0f, 0.166922331f, 0.726000845f, 0.0578664057f, 0.288887084f, 0.264229447f, 0.177596867f,
    -0.00216581812f, 0.129948959f, 0.224161908f, -0.00487308204f
};
const std::vector<float> kMlpEpsilonAlpha2Beta1 = {
    -0.415450871f, 0.0f, 0.166922331f, 0.726000845f, 0.0578664057f, 0.288887084f, 0.264229447f, 0.177596867f,
    -0.00216581812f, 0.129948959f, 0.224161908f, -0.00487308204f
};
const std::vector<float> kMlpEpsilonGammaBox = {
    -0.415450871f, 0.0f, 0.166922331f, 0.726000845f, 0.0578664057f, 0.288887084f, 0.264229447f, 0.177596867f,
    -0.00216581812f, 0.129948959f, 0.224161908f, -0.00487308204f
};
const std::vector<float> kCnnEpsilon1e6 = {
    0.0f, 0.0f, 3.16066265f, 1.91232347f, -1.3498764f, 0.0f, -0.0890544876f, 0.406214774f, -1.22019601f,
    3.06222343f, 1.54673386f, 0.149986088f, -0.362466574f, 0.165608749f, -2.65288734f, -2.12481189f, -1.15770662f,
    -0.656189501f, 0.0f, -3.86057925f, 0.895225585f, 0.668686628f, 0.121863186f, -0.170296878f, -0.837430716f,
    0.554640114f, -1.96856558f, -1.83576894f, 1.23894882f, 1.78733611f, -0.0289033558f, 3.10283089f, 3.31219053f,
    -0.0187476873f, -3.14424253f, 0.0f, -0.404530197f, -0.088179633f, -0.155933544f, -0.169633642f, 0.0457089469f,
    -0.0836959034f, 0.117572829f, -0.0236641429f, 0.422838777f, 0.0316350907f, 0.0508155748f, 0.0453353003f,
    0.568683684f, 0.0797104388f, 0.310496926f, 0.283345401f, 0.0f, -0.294679493f, -0.214097098f, -0.344747484f,
    -0.333288968f, 0.0145720523f, 0.642665267f, 0.118071064f, -0.260304213f, 0.313859761f, -0.201766908f,
    -0.626722813f, -0.0107110757f, -0.044837065f, 0.0f, 0.238383889f, 0.731467426f, 0.0f, 0.130774751f, 0.0f
};
const std::vector<float> kCnnEpsilon025 = {
    0.0f, 0.0f, 2.55147958f, 1.10956371f, -0.796041131f, 0.0f, -0.0634988844f, 0.710042953f, -0.479068696f,
    1.91348255f, 0.962274075f, 0.0741757303f, -0.262576222f, -0.00427999813f, -1.56489885f, -1.50646305f,
    -0.712808669f, -0.351817816f, 0.0f, -2.27903581f, 0.151966184f, 0.194114253f, 0.0155711174f, -0.10084866f,
    -0.950804412f, 0.436012387f, -0.995702326f, -1.06631768f, 0.735300481f, 1.04772031f, -0.00607261807f,
    1.31696975f, 1.67542863f, 0.031917274f, -1.83618844f, 0.0f, -0.291936904f, -0.0676607192f, -0.129727855f,
    -0.119934104f, 0.0355494693f, -0.0587229654f, 0.0835928842f, -0.0251600351f, 0.327636838f, 0.0292638093f,
    0.0795862079f, 0.035098508f, 0.393925846f, 0.0597003251f, 0.236698225f, 0.197933555f, 0.0f, -0.241964936f,
    -0.149268717f, -0.26509279f, -0.240285039f, 0.00982990675f, 0.51660645f, 0.0941432565f, -0.181967974f,
    0.250318736f, -0.147290379f, -0.434380978f, -0.00664212741f, -0.028101325f, 0.0f, 0.177608073f, 0.550785244f,
    0.0f, 0.081962198f, 0.0f
};
const std::vector<float> kCnnZPlus = {
    0.0f, 0.0f, 0.00132812641f, 0.00114140159f, 0.00183747057f, 0.0f, 0.0f, 0.0113752065f, 0.00618584454f,
    0.0095534455f, 0.00771504827f, 0.00135203172f, 0.130388677f, 0.00424713176f, 0.113996752f, 0.042531155f,
    0.00863579661f, 0.0048888796f, 0.0f, 0.0721139312f, 0.0492003784f, 0.043023847f, 0.0326055773f, 0.00190306955f,
    0.149320349f, 0.0313498676f, 0.087970823f, 0.0632235184f, 0.0f, 0.0f, 0.0f, 0.0227789953f, 0.00585689489f,
    0.00999111496f, 0.0166518576f, 0.0f, 0.00265531335f, 0.00228199572f, 0.00702134054f, 0.0152600743f,
    0.00548660383f, 0.0f, 0.00367364101f, 0.015531362f, 0.0759529769f, 0.0258495193f, 0.116531014f, 0.00397052523f,
    0.0282119885f, 0.0353132486f, 0.0429246724f, 0.0729939118f, 0.0f, 0.0f, 0.00324200094f, 0.0329361036f,
    0.106196105f, 0.00458617136f, 0.120412163f, 0.0238231495f, 0.00774683198f, 0.0293861702f, 0.0219796151f,
    0.0627989024f, 0.0f, 0.0f, 0.0f, 0.0136547694f, 0.0227579493f, 0.0f, 0.0f, 0.0f
};
const std::vector<float> kCnnAlphaBeta21 = {
    0.0f, 0.0f, -0.330740631f, -0.154153824f, 0.104033723f, 0.0f, 0.00159902754f, -0.333980441f, 0.0568987243f,
    -0.116607808f, -0.139441252f, -0.00681822188f, 0.809518039f, 0.083798714f, 0.533917308f, 0.264545053f,
    0.072969988f, 0.030202359f, 0.0f, 0.709632993f, -0.00981198251f, -0.299249589f, 0.093447268f, 0.0217309874f,
    0.887568951f, -0.200054884f, 0.617224336f, 0.358572721f, -0.13324599f, -0.152654946f, 0.00190938753f,
    -0.893926084f, -0.82969749f, -0.137435436f, 0.278907627f, 0.0f, -0.204436868f, -0.0767863616f, -0.277384698f,
    -0.16644603f, 0.0577402525f, -0.122574478f, 0.0632565841f, -0.0766278207f, 0.475019246f, 0.213426918f,
    0.356711537f, 0.0900243372f, 0.205092594f, 0.0651635826f, 0.266035229f, -0.814521074f, 0.0f, -0.845039248f,
    -0.0914117247f, -0.391771734f, 0.0131320506f, -0.0876707733f, 1.47850752f, 0.194427818f, -0.107405372f,
    0.417237818f, -0.195065781f, -0.231259465f, -0.00789420307f, -0.00948496535f, 0.0f, 0.225368023f, 0.492704511f,
    0.0f, 0.0168114919f, 0.0f
};
const std::vector<float> kCnnGamma025 = {
    0.0f, 0.0f, 0.779430866f, 0.386208534f, -0.195293963f, 0.0f, -0.0155207058f, 0.309043556f, -0.0436733961f,
    0.683716297f, 0.344118565f, 0.0172683112f, -0.0587219857f, 0.00204412313f, -0.126455829f, -0.376893252f,
    -0.198534489f, -0.0751127526f, 0.0f, -0.336389363f, -0.0347919613f, -0.0550725088f, 0.015480645f,
    -0.0331158638f, -0.296870589f, 0.0681059361f, -0.23255983f, -0.294024199f, 0.188109368f, 0.274326771f,
    -0.000435719441f, 0.164567232f, 0.316194922f, -0.0183119476f, -0.479226708f, 0.0f, -0.138342291f,
    -0.0371769778f, -0.0821971595f, -0.0494460948f, 0.0240601674f, -0.0276260152f, 0.0415148437f, -0.0250096843f,
    0.240283847f, 0.0275887009f, 0.14885442f, 0.0201747082f, 0.193375781f, 0.0567783713f, 0.139849707f,
    0.111314677f, 0.0f, -0.154651478f, -0.0731767491f, -0.115223393f, -0.062589407f, 0.00791420508f, 0.363166511f,
    0.0723056346f, -0.0814826488f, 0.158336625f, -0.0700751245f, -0.20250994f, -0.0038725303f, -0.0161824021f, 0.0f,
    0.109614134f, 0.301623732f, 0.0f, 0.0377589352f, 0.0f
};
const std::vector<float> kCnnEpsilonPlus = {
    0.0f, 0.0f, 0.930260658f, 0.55911696f, -0.0126747098f, 0.0f, 0.0f, 0.91716373f, 0.309825689f, 1.16578782f,
    0.589297235f, 0.000548684038f, -0.697551906f, 0.0179781877f, 0.451055408f, 0.056704998f, 0.0687166974f,
    0.0459013991f, 0.0f, -0.427404463f, -0.34757176f, -0.300751716f, -0.0782184377f, -0.0229316875f, -1.16552544f,
    -0.26458928f, -0.812300444f, -0.589045644f, 0.0851548612f, 0.170309722f, 0.0f, -0.274483323f, -0.070574671f,
    -0.104424819f, -0.200652257f, 0.0f, -0.0332552567f, -0.018665269f, -0.00296906172f, -0.00484268228f,
    0.0105444398f, 0.0f, 0.00686872005f, -0.0139891692f, 0.121393837f, 0.0324097574f, 0.197920799f, 0.00763075985f,
    0.0434220284f, 0.0449458659f, 0.0734454021f, 0.123055756f, 0.0f, 0.0f, -0.00698679592f, 0.0215479638f,
    0.0145752151f, 0.00881394092f, 0.188150942f, 0.0457845591f, -0.00500698574f, 0.0427076928f, -0.0105352886f,
    -0.0654948354f, -0.00225861557f, -0.00903446227f, 0.0f, 0.022807179f, 0.0437374003f, 0.0f, 0.0f, 0.0f
};
const std::vector<float> kCnnEpsilonAlpha2Beta1 = {
    0.0f, 0.0f, 4.21546125f, 2.07582903f, -1.27230191f, 0.0f, -0.0285505876f, 4.81258631f, -0.426498711f,
    1.68262172f, 2.01066947f, 0.0530080311f, -3.37554359f, -0.942018747f, -0.877755463f, -2.44505763f, -1.30026031f,
    -0.214389727f, 0.0f, -7.22416544f, -1.25096667f, 0.982547939f, -0.411693454f, -0.206194431f, -3.98288202f,
    1.4490031f, -3.23327947f, -2.31203938f, 1.1571362f, 1.51372242f, -0.0115039106f, 5.38583469f, 4.99886322f,
    0.558501363f, -2.63273811f, 0.0f, -0.189579815f, -0.0700723082f, -0.264048934f, -0.154553801f, 0.0549586229f,
    -0.117784977f, 0.0593119003f, -0.0716771185f, 0.451602101f, 0.208901614f, 0.348247111f, 0.0865067169f,
    0.195429519f, 0.0629669428f, 0.232319146f, -0.792106509f, 0.0f, -0.812020063f, -0.0857896805f, -0.36486873f,
    0.0396783873f, -0.0875668079f, 1.42155766f, 0.18668057f, -0.0998354256f, 0.390008897f, -0.175562099f,
    -0.197724491f, -0.00727728801f, -0.00874374062f, 0.0f, 0.211932778f, 0.462501109f, 0.0f, 0.0154977208f, 0.0f
};
const std::vector<float> kCnnEpsilonGammaBox = {
    0.0f, 0.00722560659f, 0.850848198f, 0.895956576f, 0.0157970041f, 0.0f, 0.0387376025f, 0.522368908f,
    0.572281122f, 1.22101855f, 0.483450711f, 0.00669220276f, -0.5972718f, 0.165360197f, 0.0623226911f, 0.230112225f,
    0.410103112f, -0.0754417926f, -0.442875504f, -0.433934659f, -0.159438297f, -0.145783529f, -0.0318472199f,
    -0.238601282f, -0.818791389f, -0.341572285f, -0.532561541f, -0.849700689f, -0.071398139f, 0.165592328f,
    0.000952928618f, 0.127557442f, -0.0779367834f, -0.712275326f, -0.412780762f, 0.14031373f, -0.0395284072f,
    -0.0416604765f, 0.0220774226f, -0.0212289356f, 0.0372567698f, -0.00623361859f, 0.011635717f, -0.0303699262f,
    0.129239932f, 0.0427116007f, 0.117836595f, 0.0134609565f, 0.034241572f, 0.00367869437f, 0.0963536575f,
    0.0820051432f, 0.0675008148f, 0.000404573977f, -0.0358963348f, 0.00377751514f, -0.0197151341f, 0.0491787568f,
    0.127631843f, 0.075514935f, -0.0379713699f, 0.104178384f, -0.00232800469f, -0.0449568182f, -0.0049035009f,
    -0.00467378274f, 0.0f, 0.0934216231f, 0.0884160399f, -0.0440972559f, -0.00155792758f, 0.0f
};
// ---- Part B: LXT 2.1 explicit AttnLRP ----
const std::vector<float> kMhaLogits = {
    -2.57677889f, 0.9632231f, 1.37123692f, -0.579155743f, -3.64022136f, 3.72176266f
};
const std::vector<float> kMhaRelevance = {
    0.0319217965f, 0.00072420761f, 0.463613153f, -0.171174243f, 0.00498327613f, -0.125981301f, 0.202886865f, 0.0f,
    0.263726115f, -0.123315476f, -0.104637057f, -0.00127870962f, 0.245525748f, 0.180855125f, 0.0354556926f,
    -0.0351926908f, -0.0953235254f, 0.00855000876f, 0.0162152797f, -0.0248715617f, -0.10872291f, 0.0187298283f,
    0.0247559696f, 0.0587989911f
};
const std::vector<float> kBlockLogits = {
    -5.69135284f, -7.39651966f, 25.0383968f, 8.39717293f, -17.4273224f, 5.72946692f
};
const std::vector<float> kBlockRelevance = {
    -0.0306345634f, -0.079576768f, -0.26154691f, -0.220140398f, -0.213136375f, 1.00798154f, 0.329797894f, 0.0f,
    -0.219709605f, 0.575962424f, 0.00949888676f, -0.0991709232f, -0.125010744f, -0.0847015977f, -0.0113318935f,
    0.160036296f, 0.470930874f, -0.00201122975f, -0.131313622f, 0.33547467f, 0.815548956f, 0.160692096f,
    -0.495223135f, -0.358061969f
};

struct ReferenceCase {
    const char* name;
    LRP lrp;
    const std::vector<float>* expected;
};

// Zennit's uniform rules (LayerMapComposite, Pass on ReLU, one rule on every Linear/Conv2d) and
// its three presets, for the given ZBox box.
std::vector<ReferenceCase> ZennitCases(const std::vector<float>& eps_1e6, const std::vector<float>& eps_025,
                                       const std::vector<float>& zplus, const std::vector<float>& alpha_beta,
                                       const std::vector<float>& gamma, const std::vector<float>& epsilon_plus,
                                       const std::vector<float>& epsilon_alpha2_beta1,
                                       const std::vector<float>& epsilon_gamma_box) {
    return {
        {"Epsilon(1e-6)", LRP(ZennitEpsilon(1e-6f)), &eps_1e6},
        {"Epsilon(0.25)", LRP(ZennitEpsilon(0.25f)), &eps_025},
        {"ZPlus", LRP(AlphaBetaRule(1.0f, 0.0f)), &zplus},
        {"AlphaBeta(2,1)", LRP(AlphaBetaRule(2.0f, 1.0f)), &alpha_beta},
        {"Gamma(0.25)", LRP(GammaRule(0.25f)), &gamma},
        {"EpsilonPlus", LRP::epsilon_plus(), &epsilon_plus},
        {"EpsilonAlpha2Beta1", LRP::epsilon_alpha2_beta1(), &epsilon_alpha2_beta1},
        {"EpsilonGammaBox(-3,3)", LRP::epsilon_gamma_box(-3.0f, 3.0f), &epsilon_gamma_box},
    };
}

// ---- Part A: Zennit ------------------------------------------------------------------------------

// Linear(6,8) -> ReLU -> Linear(8,5) -> ReLU -> Linear(5,3), biases nonzero, input (2, 6) with
// mixed signs (and one exact zero), targets {2, 1}.
TEST(LRPReferenceTest, MlpMatchesZennit) {
    CPUBackend backend;
    LinearModule l1(6, 8, &backend), l2(8, 5, &backend), l3(5, 3, &backend);
    SetLinear(l1, 6, 8, 5, 1, 7, 2, 11);
    SetLinear(l2, 8, 5, 3, 1, 9, 6, 13);
    SetLinear(l3, 5, 3, 13, 1, 6, 10, 7);
    ReluModule r1(&backend), r2(&backend);
    ExplainerContext ctx({&l1, &r1, &l2, &r2, &l3});
    const Tensor x(Shape({2, 6}), &backend, Det(12, 5, 3, 5));

    for (ReferenceCase& c : ZennitCases(kMlpEpsilon1e6, kMlpEpsilon025, kMlpZPlus, kMlpAlphaBeta21, kMlpGamma025,
                                        kMlpEpsilonPlus, kMlpEpsilonAlpha2Beta1, kMlpEpsilonGammaBox)) {
        SCOPED_TRACE(c.name);
        Attribution a = c.lrp.explain(ctx, x, LRPTarget{{2, 1}, {}, LRPSeed::OneHot}, &backend);
        ExpectMatchesReference(a.values, *c.expected);
    }
}

// Conv2d(1,2,3) -> ReLU -> Conv2d(2,3,2) -> ReLU -> Flatten -> Linear(27,4), biases nonzero,
// input (2,1,6,6) in [-8/3, 8/3] (inside EpsilonGammaBox's [-3, 3] box), targets {1, 3}.
TEST(LRPReferenceTest, CnnMatchesZennit) {
    CPUBackend backend;
    Conv2DModule c1(1, 2, 3, 3, &backend), c2(2, 3, 2, 2, &backend);
    SetConv(c1, 1, 2, 3, 5, 2, 6, 4, 9);
    SetConv(c2, 2, 3, 2, 7, 5, 8, 7, 10);
    ReluModule r1(&backend), r2(&backend);
    FlattenModule flatten(&backend);
    LinearModule fc(27, 4, &backend);
    SetLinear(fc, 27, 4, 11, 3, 13, 1, 7);
    ExplainerContext ctx({&c1, &r1, &c2, &r2, &flatten, &fc});
    const Tensor x(Shape({2, 1, 6, 6}), &backend, Det(72, 7, 1, 3));

    for (ReferenceCase& c : ZennitCases(kCnnEpsilon1e6, kCnnEpsilon025, kCnnZPlus, kCnnAlphaBeta21, kCnnGamma025,
                                        kCnnEpsilonPlus, kCnnEpsilonAlpha2Beta1, kCnnEpsilonGammaBox)) {
        SCOPED_TRACE(c.name);
        Attribution a = c.lrp.explain(ctx, x, LRPTarget{{1, 3}, {}, LRPSeed::OneHot}, &backend);
        ExpectMatchesReference(a.values, *c.expected);
    }
}

// ---- Part B: LXT AttnLRP -------------------------------------------------------------------------

void SetAttention(MultiHeadAttentionModule& mha) {
    SetLinear(mha.q_proj(), 4, 4, 5, 1, 6, 2, 11);
    SetLinear(mha.k_proj(), 4, 4, 7, 3, 5, 5, 13);
    SetLinear(mha.v_proj(), 4, 4, 11, 6, 7, 8, 9);
    SetLinear(mha.out_proj(), 4, 4, 13, 2, 8, 1, 10);
}

std::vector<float> PlusOne(std::vector<float> v) {
    for (float& x : v) x += 1.0f;
    return v;
}

// Runs <body> -> Flatten -> Linear(12, 3) under LXT's AttnLRP settings and checks both the
// logits (same network) and the input relevance against LXT. `backend` must be body's (body
// caches the input tensor, so that tensor's backend has to outlive body).
void ExpectAttentionMatchesLxt(CPUBackend& backend, Module& body, const std::vector<float>& logits,
                               const std::vector<float>& relevance) {
    FlattenModule flatten(&backend);
    LinearModule head(12, 3, &backend);
    SetLinear(head, 12, 3, 11, 5, 9, 3, 7);
    ExplainerContext ctx({&body, &flatten, &head});
    const Tensor x(Shape({2, 3, 4}), &backend, Det(24, 5, 7, 4));

    ExpectMatchesReference(ctx.forward_pass(x), logits);
    Attribution a = LRP(ZennitEpsilon(1e-6f)).explain(ctx, x, LRPTarget{{0, 2}, {}, LRPSeed::OneHot}, &backend);
    ExpectMatchesReference(a.values, relevance);
}

// MultiHeadAttentionModule(d_model 4, 2 heads, no RoPE / QK-Norm) on (2, 3, 4), targets {0, 2}.
// LXT ops: linear_epsilon on Q/K/V/O, matmul (epsilon + uniform 1/2 split) for Q@K^T and
// Attn@V, softmax with temperature sqrt(head_dim) (AttnLRP Prop. 3.1 on the scaled scores).
TEST(LRPReferenceTest, MultiHeadAttentionMatchesLxtAttnLRP) {
    CPUBackend backend;
    MultiHeadAttentionModule mha(4, 2, &backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    SetAttention(mha);
    ExpectAttentionMatchesLxt(backend, mha, kMhaLogits, kMhaRelevance);
}

// Pre-norm TransformerBlock(d_model 4, 2 heads, d_ff 6, no RoPE / QK-Norm): adds LXT's
// rms_norm_identity (both norms), add2 (both residuals: epsilon split), and the SwiGLU rules
// (identity on SiLU, mul2's uniform 1/2 split of gate * up, linear_epsilon projections).
TEST(LRPReferenceTest, TransformerBlockMatchesLxtAttnLRP) {
    CPUBackend backend;
    TransformerBlock block(4, 2, 6, &backend, /*use_rope=*/false, /*use_qk_norm=*/false);
    block.norm1().set_gamma(PlusOne(Det(4, 3, 14, 8)));
    block.norm2().set_gamma(PlusOne(Det(4, 5, 11, 8)));
    SetAttention(block.mha());
    SetLinear(block.swiglu().gate_proj(), 4, 6, 5, 9, 6, 4, 12);
    SetLinear(block.swiglu().up_proj(), 4, 6, 7, 2, 7, 7, 10);
    SetLinear(block.swiglu().down_proj(), 6, 4, 3, 8, 6, 9, 11);
    ExpectAttentionMatchesLxt(backend, block, kBlockLogits, kBlockRelevance);
}

}  // namespace
}  // namespace pulsatrix
