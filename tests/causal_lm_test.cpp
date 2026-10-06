// IO-4: Hugging Face checkpoints into a pulsatrix CausalLM by name. tests/fixtures/hf_tiny holds tiny
// randomly initialized Llama, Qwen2 and Qwen3 models saved by transformers' own save_pretrained
// (bf16, transformers 5 config format, one sharded); the reference logits and greedy generations
// come from transformers itself (tools/generate_hf_tiny_models.py).

#include "pulsatrix/causal_lm.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include "pulsatrix/cpu_backend.hpp"
#include "pulsatrix/lrp_rule_config.hpp"

namespace pulsatrix {
namespace {

// ---- Reference values: tools/generate_hf_tiny_models.py output, verbatim ----
const std::vector<float> kLlamaLogits = {
    -1.65257883f, 0.146860853f, 0.672399521f, 2.2642417f, -0.350977659f, 0.775681257f, -0.397817075f, -0.794840872f,
    0.890354395f, 1.64442849f, 1.26065457f, -1.04628623f, -0.0478919521f, 0.541087091f, -0.00918882154f,
    -2.18642735f, 0.298877418f, 0.0711255893f, -1.36446559f, 0.450472355f, -0.79552418f, -0.0395605825f,
    1.07370496f, 1.66397333f, -0.0227888133f, 0.254664391f, -0.20484674f, -0.843975067f, 0.0994973108f, 2.20474887f,
    0.358769685f, -0.139121071f, -0.107061431f, 0.250680476f, 0.230137408f, -0.122134514f, 0.355304003f,
    1.35203445f, 0.736076653f, 0.174148917f, -1.87706959f, 0.521118104f, 1.65559673f, 2.36202574f, -0.912577927f,
    -0.508586347f, -0.548395097f, 0.00205293717f, -1.26266432f, -0.275078565f, -0.951628983f, -1.58268881f,
    2.10232353f, 0.726811528f, 1.48383498f, 0.437964022f, 0.610505521f, -1.8076216f, 1.90882611f, -0.494112104f,
    0.805787981f, 0.0140388794f, 0.915976167f, 2.01093912f, 0.711474895f, 2.08067203f, -0.506004453f, -0.630644798f,
    1.31566489f, 0.453619182f, 1.60517275f, -1.98672342f, -0.0745855197f, -0.590028346f, 0.61569798f, 0.451863527f,
    0.39332968f, -0.003701051f, -0.501964867f, 0.791021526f, -0.800254643f, 0.407114983f, -0.635181665f,
    1.90226698f, 0.925096214f, -2.02727532f, -0.0340092145f, -2.76053262f, -0.937438905f, -1.13900363f,
    -1.95899296f, -2.01138306f, 0.420849144f, 1.0330174f, 1.5484668f, -2.46475744f, -0.70281446f, -0.407871306f,
    1.12364018f, 0.671084464f, 0.921599448f, 0.788843036f, 0.453591079f, -0.257347852f, -0.0838359594f,
    0.198850855f, 2.23711038f, 1.28529251f, -1.06914639f, 2.72559667f, -1.13261521f, -0.67053467f, -1.74236083f,
    -0.0386689492f, 0.841552675f, 0.989226699f, -0.956868351f, -0.363938689f, 1.12083375f, -0.973283172f,
    0.353937149f, -0.976329267f, 1.31164837f, -1.0138638f, 1.71020842f, -0.306525558f, 2.08038807f, 1.51807559f,
    0.188910589f, 2.29456306f, 0.0253523849f, 0.841023147f, -0.270226032f, 3.33648062f, 2.25797296f, 1.00583339f,
    0.815289557f, -0.0286934935f, 0.587480128f, 1.05843675f, 0.557509601f, -0.35224089f, -1.07375801f, 0.20963721f,
    2.18807459f, 0.803223729f, -0.376909316f, -0.312620848f, 0.378822565f, -1.25142515f, -0.414616317f,
    -0.441977501f, -0.401632547f, -0.76700902f, -1.25686085f, -1.48536742f, -1.81837618f, 1.11446452f, 0.226414338f,
    -0.893171191f, -2.54817176f, -0.355835348f, -1.19556642f, -1.32175577f, 1.04573596f, 0.468352437f, 1.50294673f,
    -0.1479339f, -0.678795874f, 1.17929757f, 3.96271682f, 0.595759034f, -1.02277398f, 0.66755873f, -0.408067077f,
    1.63525808f, -2.4282949f, 0.348836154f, 1.2073015f, -0.674996555f, -0.391274959f, -0.130689606f, -0.332141012f,
    1.14238584f, 0.00677352073f, -1.1723299f, 0.10276708f, -1.52935874f, -0.335909814f, -0.059794683f,
    -0.560478032f, -0.0664473772f, 0.860668898f, 1.58533025f, 0.404890209f, 1.43827415f, -1.00223172f, 2.94875216f,
    3.44594193f, 0.698388278f, 0.325917751f, 0.0535748452f, 0.552900672f, 1.6627506f, -0.252522826f, 0.223632932f,
    -1.85916412f, -0.0223320927f, 2.00726366f, 1.14375865f, -0.203283682f, 0.0334546231f, 1.1854471f,
    -0.0155264642f, -0.456687659f, 1.64275301f, 0.319662154f, -1.49266899f, -0.544339359f, -0.295249075f,
    -1.16703939f, 1.0130558f, 0.789970577f, -0.0143440133f, -1.44524908f, -0.706642151f, -0.774064481f,
    -2.39659119f, 0.819055736f, -0.683036506f, 0.0277680699f, 0.144980818f, 0.0652128607f, 0.358285964f,
    4.17444134f, -1.32933569f, -0.525906265f, 0.109084748f, 0.290921956f, 1.65665615f, -1.23874509f, 0.946951866f,
    0.28113541f, 0.146383107f, -0.338323861f, -1.40593374f, -0.814578116f, 0.621168017f, -0.387502491f,
    -1.00819433f, -1.17634451f, -1.78581357f, -0.165299371f, 1.34046566f, -0.778259158f, -0.582979858f,
    0.458634466f, -0.582586527f, -1.1724267f, 0.0058940053f, 0.102595121f, 0.423186004f, -0.479470849f,
    -0.0833848715f, -0.679663718f, 0.393837154f, 1.92894936f, 0.818491399f, -1.29381144f, -0.896292388f,
    -3.53652382f, -1.89724672f, 1.77405834f, 2.11791706f, -0.618246019f, 0.490667701f, -0.86534971f, 0.774194539f,
    0.117141128f, 1.69557357f, 0.00479745865f, -1.81327784f, 1.15099657f, 0.574344575f, 0.675683618f, -0.176310286f,
    -0.75810504f, -0.326262593f, -0.889763057f, 1.33090162f, -0.454500496f, -1.26004171f, 0.733720303f,
    -2.00185466f, -0.515675187f, 0.36469838f, -0.721489012f, -0.59464252f, 0.106412292f, -1.22058463f,
    -0.717007756f, -0.369389296f, 0.138574511f, 0.292631626f, -0.116877377f, 0.030064404f, 1.31878924f,
    -1.49766922f, -2.31749368f, -0.983351409f, 0.143777087f, 0.599385977f, -0.137840807f, -0.233108431f,
    0.469932944f, -0.0468766391f, 0.653826058f, 0.388194352f, 1.03690958f, 0.248761922f, 0.202893317f, 0.73352313f,
    -0.437991858f, -0.0587781072f, -0.86058569f, 1.12501371f, 2.30175614f, 0.112507701f, 0.575750411f, 0.239232302f,
    1.0466696f, 0.851265073f, -1.24262166f, -0.276497006f, -2.37053204f, 0.314417839f, 1.94615364f, 0.189155281f,
    -0.0469556749f, 1.21366954f, 1.5425818f, -0.299668968f, -0.689904094f, 0.21254164f, 0.437285542f, -0.827774346f,
    0.563835382f, -0.365132064f, -0.695659637f, 1.28995061f, 0.89669019f, -0.472114086f, -1.7194612f,
    -0.0489193797f, -0.96875f, -2.578758f, 1.04973984f, -0.734055877f, -0.0717101991f, 0.475229889f, -0.0558369756f,
    0.681343913f, 4.23448467f, -0.983738661f, -0.423573107f, -0.395667255f, 0.585318983f, 1.08422649f, -1.45040619f,
    1.26783049f, 0.390328795f, 0.450485766f, 0.579145312f, -0.563952267f, -0.330305934f, -0.195519924f,
    -0.979249358f, 0.0121928453f, -0.505752802f, -2.43331861f, 0.674247622f, 0.900463641f, 0.0595864654f,
    -0.494140446f,
};
const std::vector<int64_t> kLlamaGenerated = {42, 41, 15, 6, 29, 45, 29, 50};
const std::vector<float> kQwen2Logits = {
    0.565614998f, 0.437000364f, 0.897490919f, -1.0340991f, -1.14950657f, -2.29266381f, 2.11730361f, 0.239208609f,
    0.130841702f, -1.74379635f, 0.333236426f, -1.17880869f, -0.0791057944f, 0.200417221f, 0.374896735f,
    0.761094332f, -0.619895458f, -0.121129885f, 1.5205065f, -0.888660431f, -2.31621122f, 0.569305122f, 2.12321162f,
    -1.53446782f, 0.0157494973f, 2.33955288f, -1.56496048f, -0.858078122f, -0.495133996f, -0.221941158f,
    -2.61522269f, 0.36675477f, 0.152258098f, 0.542758703f, -0.827343941f, 0.034311384f, 0.655638456f, -0.564432561f,
    0.367572039f, 1.51962233f, 0.343330115f, 1.09060359f, 1.40125597f, -1.75226653f, 2.0175283f, 0.259507656f,
    1.09808695f, -1.8365376f, -1.79487264f, 1.42747462f, 1.41153133f, 0.28291446f, 0.284938604f, 1.41966856f,
    -0.199788421f, 1.706231f, -0.20019047f, -0.0998401493f, 1.447263f, 0.610220015f, 0.523479998f, 1.82755554f,
    -1.18883836f, 1.06656265f, 0.689147711f, 1.61623693f, 1.13568437f, -0.708615065f, -2.04776645f, -2.89925766f,
    0.861844718f, -1.94698536f, -0.921122789f, -0.963466167f, 0.226323649f, -2.99650002f, -1.46386588f,
    0.667513072f, -0.376875907f, 0.956535876f, -0.153672412f, -0.838486075f, 0.707786679f, -0.982139349f,
    -3.91327763f, -0.216847643f, 2.2015636f, 0.235748649f, -0.496091247f, 0.714563131f, -1.12750828f, -1.20659471f,
    0.432211131f, 1.58117437f, -1.48170185f, -1.29122782f, -0.590839624f, 1.02116299f, -1.31954134f, -0.0448590703f,
    1.21706271f, -0.403077781f, -0.461731464f, 0.606897473f, 1.60725367f, 0.658535719f, 1.04024708f, -0.501468956f,
    1.33352625f, 1.03078949f, 0.924178779f, -1.11396432f, 0.491070002f, 0.653215706f, 0.799511254f, 0.396891952f,
    0.288693279f, 0.835604846f, 0.37871474f, 0.581825078f, -0.881391525f, -0.0392039344f, 2.22561407f, 1.80903518f,
    1.64709818f, 1.19087136f, -0.806324184f, 0.340310901f, 0.889512479f, 1.23413002f, 1.00765073f, -0.240061268f,
    -1.64605546f, 0.107772134f, -1.1099987f, 0.597373486f, 0.97351706f, 0.366252124f, 1.41760886f, 1.21735084f,
    1.20845628f, -0.20836775f, 0.101353899f, -1.37800336f, -0.455985606f, 0.506279469f, 0.109425537f, -2.25788569f,
    -2.3755753f, 0.0373466052f, 0.26638326f, -1.29881442f, 1.48221338f, 0.898001313f, 0.428549379f, 0.855868459f,
    0.375738889f, -0.568410695f, -0.408960521f, -1.65994263f, 0.109061122f, 2.95258427f, 0.54932642f, 0.570106566f,
    0.849445939f, -1.12166417f, -0.151535437f, -0.0609081276f, 0.333462656f, 1.23103118f, 1.22901237f,
    -0.797539949f, -0.694850385f, 0.449078798f, 1.17897344f, -0.772392154f, 0.684454322f, -0.727200449f,
    -1.22885859f, -1.99222291f, -1.02158737f, -1.03087211f, 0.928690732f, 1.39095557f, 0.715672791f, 0.691954255f,
    3.01169348f, -1.25435197f, -0.355629593f, 0.327936769f, 0.494496197f, -0.138556093f, -0.355699003f, 1.26792371f,
    0.832880914f, -0.0677672774f, -0.713948846f, -2.7954011f, -1.28083146f, -2.44099379f, -1.0552516f, 0.693600416f,
    -0.0685111061f, 0.445801437f, -0.876669228f, -0.527364194f, 0.270123899f, 0.902830899f, -0.146277532f,
    0.771534383f, 0.989312232f, -0.175458446f, -2.83867979f, -1.55009842f, 0.059326157f, 0.646550357f,
    -0.0488155112f, -0.381106555f, -1.46055853f, -2.09656787f, -0.361635834f, 1.74877036f, -0.8699947f,
    -1.57500315f, 0.611685336f, 1.34591162f, 0.044692792f, 0.628937304f, -0.252472103f, -0.921297252f, 0.515495896f,
    -0.803112149f, -0.105431803f, 0.0698581636f, 1.4710747f, -0.0997727364f, 1.10037661f, 0.569454014f, 1.34486604f,
    -0.680392861f, -0.245215043f, 1.27032638f, 0.143230781f, 0.363630742f, 0.403372943f, 0.68736577f, -0.123651937f,
    -1.12668371f, -0.244229108f, 2.25309539f, 2.80232978f, 1.22105312f, 0.692989588f, 0.647067964f, -1.27599847f,
    -0.340267271f, 1.31168222f, 1.94149745f, 0.837939262f, 0.407487959f, -1.68550706f, 0.509296179f, -1.39495516f,
    1.04249299f, -1.08972275f, -0.566868186f, 1.32464254f, -0.574983358f, -0.190818429f, -0.45891422f,
    -0.757843256f, -1.31172001f, -1.14736068f, 0.366690278f, 0.420972347f, -3.37704802f, -2.03762913f,
    -0.630560875f, 1.46836829f, -0.227120161f, 0.782778919f, 1.31964922f, 1.29969823f, 1.09344661f, 0.758856356f,
    -0.105754241f, -0.912877619f, -2.76433587f, -0.664459705f, 1.6318022f, -0.80883348f, 0.230301648f,
    0.0539252162f, -0.562554598f, 0.223844558f, -0.670829058f, 0.262854099f, 2.5362606f, -0.38791123f,
    -0.284177154f, -0.832377374f, -0.135568082f, 0.713602543f, 0.381814241f, 1.96528101f, -1.44960368f,
    -1.65650725f, -1.04610014f, 0.0349216461f, 0.0327147841f, 0.0669732094f, 0.792168379f, 0.308457434f,
    0.0466884375f, 1.67730427f, 0.576416612f, -1.12141836f, -0.923391998f, 0.673287153f, -0.158142239f,
    0.165793359f, 2.90330791f, 0.833236873f, -1.90005398f, -0.718890011f, -0.343993485f, -0.582276225f,
    -0.347513646f, 0.246107697f, 0.651440144f, 1.05880237f, 0.129039794f, -0.82128036f, 1.1820364f, -0.309274346f,
    -0.724239171f, 0.785393298f, -0.113330126f, 1.13689899f, 1.01974082f, -1.79366016f, 2.18990469f, 0.0843818486f,
    -1.75406706f, -1.02436972f, 0.29970488f, 0.0814142227f, 1.71547937f, 1.39161718f, -0.627112687f, 0.61048907f,
    -2.92238522f, -0.219725728f, 1.29054642f, 0.152308524f, -0.760317326f, 2.58322144f, -0.698133647f,
    -0.0639975816f, -0.457037181f, 0.753918946f, -1.77878284f, -0.187352747f, -0.863384247f, -1.45586753f,
    0.851965725f, 2.58188009f, -0.230958581f, 0.121744394f, -0.870307088f, 2.0003922f, -1.4623785f, -1.26584518f,
    -0.150874674f, -0.827333927f, 0.563214064f, 2.07776618f, -0.224745125f, 1.70609152f, -1.04290223f,
    -0.939535022f, -0.103276193f, 0.131880939f, 0.323616475f,
};
const std::vector<int64_t> kQwen2Generated = {1, 46, 58, 57, 20, 11, 58, 25};
const std::vector<float> kQwen3Logits = {
    1.30091107f, -0.241069987f, -0.011285481f, -1.40966392f, 0.492589921f, -0.301734686f, 0.247827634f,
    -0.672661543f, -0.71127969f, 0.524481654f, -0.585372269f, -0.46664995f, -1.81973803f, 0.504137933f, 1.31841373f,
    -1.33250046f, -0.223121822f, 2.08003545f, -0.123015933f, -0.629553616f, 0.205337688f, -0.27582556f,
    -1.29599094f, 1.94016063f, 0.109850638f, 0.223999277f, 0.392147213f, -0.34020108f, 1.19204473f, -0.106678568f,
    0.868091583f, 0.0592099577f, -1.74816668f, 0.481471568f, -2.22592568f, -0.354170501f, -1.22344065f,
    -0.253801972f, 0.295976937f, 1.23131728f, 0.554729462f, -1.57511365f, -0.294007272f, 1.41159391f, -0.446287453f,
    -0.605903924f, 3.07065034f, 0.884798169f, 0.336697936f, 1.12502086f, -1.02165949f, -0.836600482f, 1.78251052f,
    -0.247439951f, -0.161209509f, 2.24583364f, 0.39828828f, 0.453942478f, -0.382737875f, -0.0858195275f,
    0.123306155f, 0.113985203f, -1.11407328f, 0.816312313f, 1.90591443f, 0.25934431f, -0.624498725f, -1.3932085f,
    0.613028049f, 1.78096628f, -1.31171536f, 0.521167815f, 0.571525395f, 0.648355305f, -0.745330155f, -1.4807862f,
    -1.09574676f, 0.523069084f, 0.671119988f, -1.31878185f, 1.85554147f, 0.846895814f, -1.03500712f, -0.0242220163f,
    1.05186152f, 0.329738319f, -1.42259479f, 0.807817578f, -0.436517924f, -1.79940295f, -0.259604216f, 0.290993512f,
    2.7331531f, -0.390212655f, -1.03435504f, -0.877110541f, -1.34108412f, -1.14590049f, -0.0980591029f,
    0.0609303042f, -0.788707495f, -0.0858168006f, -0.154380471f, -0.562274635f, 0.523758054f, -1.70335317f,
    -0.984769166f, 2.87632155f, -0.544498503f, -1.70301616f, 2.13282442f, 0.90831238f, -0.728266716f, 1.50068069f,
    -1.06620371f, 0.611905694f, -0.286521047f, 1.8544265f, -0.19148393f, -0.0469255075f, 0.284896374f,
    -0.163396969f, 1.29701257f, -0.491063714f, 0.801497161f, -1.50691617f, -1.00215948f, 1.30229187f, 1.77487171f,
    -0.469834119f, 0.160158306f, -1.61171198f, -0.00631151861f, 0.300854266f, -0.772402346f, -0.782371998f,
    0.879391909f, 2.0535655f, -0.258601904f, -0.890507221f, -0.337210238f, 0.41893658f, 0.374948591f, -0.701737165f,
    1.12641895f, -0.942947805f, -0.82350111f, -0.0599549338f, 1.76828003f, -0.098014988f, -2.01096821f, 1.57618606f,
    -1.38540447f, -1.37627196f, -0.575443566f, 0.773911059f, 1.46291089f, -0.685916781f, -1.51875377f, -1.52203834f,
    -1.04852927f, 0.435358256f, -1.85265613f, 0.737884045f, -0.187693134f, 0.0850493982f, 0.20691365f,
    -0.623935223f, 2.16494584f, -1.38239038f, 1.29749715f, 1.65372622f, -0.214486912f, 0.159266621f, 1.6044544f,
    1.06311023f, 0.994202077f, 1.22846806f, -0.323124737f, -0.346127033f, -0.371146053f, 1.64372933f, -1.19646335f,
    1.1994307f, -0.00225927215f, -0.00259756995f, 0.90014255f, 0.0335745774f, 1.55142617f, -0.784098327f,
    -0.61236161f, 1.58218586f, -0.096400857f, -0.120366573f, 0.140066445f, -1.23936963f, 0.533285797f, 0.653539896f,
    -0.0612509921f, -0.5857687f, 0.580884576f, 0.836355984f, -0.236151695f, -1.24456036f, -1.18882048f,
    0.860083103f, 0.77495873f, -1.73860157f, 0.737260878f, 1.69458973f, -0.0501986407f, 0.391016424f, 0.243787229f,
    0.617678046f, -0.0983336046f, 1.66393769f, -0.94706732f, -0.784092605f, 0.172000349f, 0.36848703f, 1.79335141f,
    -0.0886807591f, 1.11565578f, 0.493260235f, -1.26154518f, -2.47915983f, -2.20233703f, -0.769155264f, -1.4482435f,
    0.761494577f, 0.0944579765f, 0.276772976f, 0.59477514f, -1.882074f, -0.497311473f, 1.62703192f, -0.739788115f,
    -0.0094625894f, 3.06302047f, 0.179442719f, -0.681663334f, 1.47913432f, -1.08219504f, -0.863797069f, 1.96316123f,
    -0.132977396f, 0.302379012f, 2.10575914f, 0.503450215f, -0.367399633f, 1.44900143f, -0.672203064f,
    0.0512911677f, -0.118578501f, -1.40033937f, 0.91532743f, 0.893865705f, 0.299136817f, -0.254676551f,
    -1.17901492f, 0.266982257f, 0.528729081f, -0.850719988f, 1.3438642f, 0.808568835f, 0.481650829f, -0.803041399f,
    -0.863633335f, -0.215113252f, 1.04876196f, 1.03499317f, -2.03954148f, 2.27166271f, 0.321781695f, -0.39594391f,
    0.304432869f, 0.794819415f, 0.950854063f, -0.16489847f, 0.593411803f, 0.799249411f, -1.81955302f, 0.675422728f,
    -0.391361415f, 2.43170214f, -0.736027181f, -0.492053151f, 0.362455249f, -1.17320955f, -2.12993383f,
    0.787252069f, 0.0582408905f, 0.055218786f, -0.298633695f, -0.789756119f, 0.309005499f, -0.0394250453f,
    -1.59918332f, -0.669710398f, 1.43728983f, 0.376179636f, -2.1382637f, 1.68834472f, 1.33787155f, -1.44996846f,
    1.12996387f, -1.88012171f, 1.35526013f, 0.390654564f, 0.926487327f, 0.81344676f, 0.0633158982f, 0.043448627f,
    0.21675095f, 2.52140141f, 0.198029488f, -0.132527888f, -0.920355439f, 0.848985076f, 1.73352122f, 0.198285818f,
    -0.907987833f, -0.583373904f, -1.81041384f, -0.899744391f, -0.150334701f, 0.41483438f, -1.57833433f,
    0.205308974f, 1.78111792f, -0.862612426f, -1.5842526f, 0.344716966f, 0.948752284f, 0.451931179f, -2.85433531f,
    0.66378808f, 0.560293734f, -0.944222093f, -0.571125984f, -1.27084386f, -1.03625405f, -1.03710938f, 2.32649851f,
    -1.61507034f, -0.391686141f, 2.41113901f, 0.39700532f, -0.00481128693f, 0.344143093f, 0.816030502f,
    -0.00750809908f, 1.21639788f, -0.553151071f, -0.166718364f, 0.683365464f, -1.04278326f, 0.707069874f,
    -1.05902171f, 0.385396242f, -0.895961225f, -1.35774028f, -0.530847311f, -0.153627753f, -2.83115911f,
    -0.100940756f, 1.08434141f, 0.510077477f, 0.678519845f, 1.04115152f, 0.655035257f, -0.137050211f, 1.01928878f,
    0.519942045f, -0.179208875f, 0.388751596f, -1.57687438f, -0.403043568f, 1.71089745f, -1.52692401f, 0.38140887f,
    -0.807162881f, -1.02067423f, 1.56267893f,
};
const std::vector<int64_t> kQwen3Generated = {26, 16, 58, 16, 58, 16, 52, 0};

std::string Model(const std::string& name) { return std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf_tiny/" + name; }

const std::vector<int64_t> kPrompt{3, 17, 42, 5, 60, 8};

class CausalLMTest : public ::testing::TestWithParam<const char*> {
protected:
    CPUBackend backend;
};

// The whole stack: config, sharded bf16 weights, the name mapping and its transposes, GQA, biases,
// QK-Norm, rotate-half RoPE, tied and untied heads. Logits within 1e-4 of transformers'.
TEST_P(CausalLMTest, LogitsMatchTransformers) {
    const std::string name = GetParam();
    const std::vector<float>& expected =
        name == "llama" ? kLlamaLogits : name == "qwen2" ? kQwen2Logits : kQwen3Logits;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Model(name), &backend);
    std::vector<float> ids(kPrompt.begin(), kPrompt.end());
    Tensor logits = model->forward(Tensor(Shape({1, 6}), &backend, ids));
    ASSERT_EQ(logits.shape(), Shape({1, 6, 64}));
    const std::vector<float> got = logits.to_host_vector();
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs(got[i] - expected[i]));
    EXPECT_LT(worst, 1e-4f) << name;
}

// Greedy generation, with the KV cache, picks the same tokens as transformers' generate().
TEST_P(CausalLMTest, GreedyGenerationMatchesTransformers) {
    const std::string name = GetParam();
    const std::vector<int64_t>& expected =
        name == "llama" ? kLlamaGenerated : name == "qwen2" ? kQwen2Generated : kQwen3Generated;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Model(name), &backend);
    GenerationConfig g;
    g.max_new_tokens = 8;
    EXPECT_EQ(Generate(model->next_token_logits(32), kPrompt, g).new_tokens, expected) << name;
}

INSTANTIATE_TEST_SUITE_P(TinyModels, CausalLMTest, ::testing::Values("llama", "qwen2", "qwen3"));

TEST(CausalLMLoadTest, ParameterNamesAndTyingFollowTheConfig) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> tied = LoadCausalLM(Model("llama"), &backend);
    std::unique_ptr<CausalLM> untied = LoadCausalLM(Model("qwen2"), &backend);
    auto names = [](CausalLM& m) {
        std::vector<std::string> n;
        for (NamedParamRef& p : m.named_parameters()) n.push_back(p.name);
        return n;
    };
    const auto tn = names(*tied), un = names(*untied);
    EXPECT_EQ(tn.front(), "embed_tokens.weight");
    EXPECT_EQ(std::count(tn.begin(), tn.end(), "lm_head.weight"), 0);
    EXPECT_EQ(std::count(un.begin(), un.end(), "lm_head.weight"), 1);
    EXPECT_EQ(std::count(un.begin(), un.end(), "layers.1.mha.q_proj.bias"), 1);
    EXPECT_EQ(std::count(tn.begin(), tn.end(), "layers.0.swiglu.gate_proj.bias"), 0);  // Llama MLPs have none
    // Every Hugging Face tensor went somewhere: 2 + 9 per layer for Llama.
    EXPECT_EQ(HuggingFaceMapping(tied->config()).size(), 2u + 2u * 9u);
}

TEST(CausalLMLoadTest, TheModelTrainsAndExplainsLikeAnyModule) {
    CPUBackend backend;
    std::unique_ptr<CausalLM> model = LoadCausalLM(Model("qwen3"), &backend);
    std::vector<float> ids(kPrompt.begin(), kPrompt.end());
    Tensor x(Shape({1, 6}), &backend, ids);
    Tensor logits = model->forward(x);
    std::vector<float> g(logits.numel(), 0.0f);
    g[5 * 64 + 7] = 1.0f;  // d logit[7] at the last position
    (void)model->backward(Tensor(logits.shape(), &backend, g));
    double norm = 0.0;
    for (NamedParamRef& p : model->named_parameters()) {
        for (float v : p.ref.grad->to_host_vector()) norm += static_cast<double>(v) * v;
    }
    EXPECT_GT(norm, 0.0);
    (void)model->forward(x);
    Tensor r = model->propagate_relevance(Tensor(logits.shape(), &backend, g), LRPRuleConfig{});
    EXPECT_EQ(r.shape(), Shape({1, 6}));  // one relevance score per input token
    for (float v : r.to_host_vector()) EXPECT_TRUE(std::isfinite(v));
}

class LoadWeightsTest : public ::testing::Test {
protected:
    CPUBackend backend;
    HfModelConfig config = ReadHfConfig(Model("qwen2") + "/config.json");
    HfCheckpoint checkpoint = HfCheckpoint::Open(Model("qwen2"));
};

TEST_F(LoadWeightsTest, StrictLoadingRefusesGapsAndLeavesTheModelUnchanged) {
    CausalLM model(config, &backend);
    std::vector<WeightMapping> m = HuggingFaceMapping(config);
    const std::vector<float> before = model.embed_tokens().weight().to_host_vector();

    std::vector<WeightMapping> partial(m.begin(), m.end() - 1);  // drop lm_head
    EXPECT_THROW((void)LoadWeights(model, checkpoint, partial), std::invalid_argument);
    EXPECT_EQ(model.embed_tokens().weight().to_host_vector(), before);  // nothing written

    // Non-strict: the gap is reported instead.
    WeightLoadReport r = LoadWeights(model, checkpoint, partial, WeightLoadOptions{false, {}});
    EXPECT_EQ(r.missing, (std::vector<std::string>{"lm_head.weight"}));
    EXPECT_EQ(r.unused, (std::vector<std::string>{"lm_head.weight"}));
    EXPECT_NE(model.embed_tokens().weight().to_host_vector(), before);

    std::vector<WeightMapping> wrong = m;
    wrong[0].transform = WeightTransform::Transpose;  // embed (64, 32) transposed no longer fits
    EXPECT_THROW((void)LoadWeights(model, checkpoint, wrong), std::invalid_argument);
    wrong = m;
    wrong.push_back(m[0]);  // mapped twice
    EXPECT_THROW((void)LoadWeights(model, checkpoint, wrong), std::invalid_argument);
    wrong = m;
    wrong[0].source = "model.embed_tokens.weights";
    EXPECT_THROW((void)LoadWeights(model, checkpoint, wrong), std::invalid_argument);
    wrong = m;
    wrong[0].target = "embedding.weight";
    EXPECT_THROW((void)LoadWeights(model, checkpoint, wrong), std::invalid_argument);
}

// A fused QKV tensor split by rows: three mappings take one slice each of the same source.
TEST_F(LoadWeightsTest, RowSlicesSplitAFusedTensor) {
    LinearModule q(32, 16, &backend, false), k(32, 8, &backend, false);
    struct Pair : Module {
        LinearModule& a;
        LinearModule& b;
        Pair(LinearModule& x, LinearModule& y) : a(x), b(y) {}
        Tensor backward(const Tensor& g) override { return g; }
        Tensor propagate_relevance(const Tensor& r, const LRPRuleConfig&) override { return r; }
        OpType op_type() const override { return OpType::Composite; }
        std::vector<NamedParamRef> named_parameters() override {
            std::vector<NamedParamRef> out;
            append_named_parameters(out, "q", a);
            append_named_parameters(out, "k", b);
            return out;
        }
        Tensor forward_impl(const Tensor& x) override { return x; }
    } pair(q, k);
    // qwen2's lm_head.weight is (64, 32): rows 0..15 as q, rows 16..23 as k.
    std::vector<WeightMapping> m{{"lm_head.weight", "q.weight", WeightTransform::Transpose, 0, 16},
                                 {"lm_head.weight", "k.weight", WeightTransform::Transpose, 16, 8}};
    (void)LoadWeights(pair, checkpoint, m, WeightLoadOptions{false, {}});
    const std::vector<float> full = checkpoint.tensor("lm_head.weight", &backend).to_host_vector();
    const std::vector<float> kw = k.weight().to_host_vector();  // (32, 8) = rows 16..23 transposed
    for (int64_t r = 0; r < 8; ++r) {
        for (int64_t c = 0; c < 32; ++c) {
            EXPECT_EQ(kw[static_cast<size_t>(c * 8 + r)], full[static_cast<size_t>((16 + r) * 32 + c)]);
        }
    }
    m[1].row_count = 60;  // past the end
    EXPECT_THROW((void)LoadWeights(pair, checkpoint, m, WeightLoadOptions{false, {}}), std::invalid_argument);
}

TEST(CausalLMLoadTest, RefusesConfigsItCantRunFaithfully) {
    CPUBackend backend;
    HfModelConfig llama32 = ReadHfConfig(std::string(PULSATRIX_TEST_FIXTURES_DIR) + "/hf/llama-3.2-1b.config.json");
    llama32.num_hidden_layers = 1;  // keep the test small; rope_scaling is what is refused
    llama32.vocab_size = 16;
    EXPECT_THROW(CausalLM(llama32, &backend), std::invalid_argument);
    EXPECT_THROW((void)LoadCausalLM("/nonexistent", &backend), std::runtime_error);
}

}  // namespace
}  // namespace pulsatrix
