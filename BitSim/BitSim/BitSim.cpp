
#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/torch.h>
#pragma warning(pop)

#include "BitBaseClient.h"
#include "Logger.h"

#include <iostream>

constexpr auto negative_sample_count = 3;

constexpr auto batch_size = 2;
constexpr auto segment_length = 160;
constexpr auto negative_count = 10;
constexpr auto n_observations = 4;
constexpr auto n_predictions = 2;


struct FeatureEncoderImpl : public torch::nn::Module
{
    FeatureEncoderImpl(int feature_size) {
        encoder = register_module("feature_encoder_cnn", torch::nn::Sequential{
            torch::nn::Conv1d{torch::nn::Conv1dOptions{1, feature_size, 10}.stride(5).padding(3).with_bias(false)},
            torch::nn::BatchNorm{feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{feature_size, feature_size, 8}.stride(4).padding(2).with_bias(false)},
            torch::nn::BatchNorm{feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{feature_size, feature_size, 4}.stride(2).padding(1).with_bias(false)},
            torch::nn::BatchNorm{feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{feature_size, feature_size, 4}.stride(2).padding(1).with_bias(false)},
            torch::nn::BatchNorm{feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{feature_size, feature_size, 4}.stride(2).padding(1).with_bias(false)},
            torch::nn::BatchNorm{feature_size},
            torch::nn::Functional{torch::relu}
        });
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: B*C*N*L
        const auto n_features = x.size(2);
        std::cout << x.sizes() << std::endl;
        x = x.reshape({ x.size(0), x.size(1), n_features * segment_length });
        std::cout << x.sizes() << std::endl;
        x = encoder->forward(x);
        std::cout << x.sizes() << std::endl;
        x = x.transpose(1, 2);
        std::cout << x.sizes() << std::endl;
        x = x.reshape({ x.size(0), 1, x.size(1), x.size(2) });
        std::cout << x.sizes() << std::endl;
        return x;
    };

private:
    torch::nn::Sequential encoder;
};
TORCH_MODULE(FeatureEncoder);


struct FeaturePredictorImpl : public torch::nn::Module
{
    FeaturePredictorImpl(int feature_size, int predictor_size) :
        predictor_size(predictor_size)
    {
        gru = register_module("feature_preditor_gru", 
            torch::nn::GRU{ 
                torch::nn::GRUOptions{ feature_size, predictor_size }.layers(1).bidirectional(false).batch_first(true) 
            }
        );
    }

    torch::Tensor forward(torch::Tensor observed_features) {
        const auto initial_hidden = torch::zeros({ 1, batch_size, predictor_size });
        auto prediction = gru->forward(observed_features, initial_hidden);
        return prediction.output;
    };

private:
    int predictor_size;
    torch::nn::GRU gru{ nullptr };
};
TORCH_MODULE(FeaturePredictor);


struct RepresentationLearnerImpl : public torch::nn::Module
{
    RepresentationLearnerImpl(void) {
        register_module("feature encoder", feature_encoder);
        register_module("feature predictor", feature_predictor);
    };

    std::tuple<double, double> forward(
        torch::Tensor observations,    // BxCxNxL (2x1x4x160)
        torch::Tensor future_positive, // BxCxNxL (2x1x1x160)
        torch::Tensor future_negative) // BxCxNxL (2x1x1x160)
    {
        auto observed_features        = feature_encoder->forward(observations);     // BxCxN (2x512x4)
        auto future_positive_features = feature_encoder->forward(future_positive);  // BxCxN (2x512x1)
        auto future_negative_features = feature_encoder->forward(future_negative);  // BxCxN (2x512x1)
        
        auto predictions = feature_predictor->forward(observed_features);

        auto accuracy = 1.0;
        auto info_nce = -0.5;

        return std::make_tuple(accuracy, info_nce);
    };
    
private:
    const int feature_size = 512;
    const int predictor_size = 256;
    FeatureEncoder feature_encoder{ feature_size };
    FeaturePredictor feature_predictor{ feature_size, predictor_size };
};
TORCH_MODULE(RepresentationLearner);


struct IntervalsSegment {
    std::array<float, segment_length> price;
    std::array<float, segment_length> vol_buy;
    std::array<float, segment_length> vol_sell;
};


class TradeDataset : public torch::data::Dataset<TradeDataset, IntervalsSegment>
{
public:
    TradeDataset(void) {}

    c10::optional<size_t> size(void) const {
        return 4;
    }

    IntervalsSegment get(size_t index) {
        logger.info("TradeDataset:get get it %d", index);

        auto a = IntervalsSegment{};
        return a;
    }

    //void reset(void) {}
    //void save(torch::serialize::OutputArchive& archive) const {}
    //void load(torch::serialize::InputArchive& archive) {}
};

int main() {
    logger.info("BitSim started");

    /*
    auto bitbase_client = BitBaseClient();

    constexpr auto symbol = "XBTUSD";
    constexpr auto exchange = "BITMEX";
    constexpr auto timestamp_start = date::sys_days(date::year{ 2019 } / 06 / 01) + std::chrono::hours{ 0 } + std::chrono::minutes{ 0 };
    constexpr auto timestamp_end = date::sys_days(date::year{ 2019 } / 06 / 01) + std::chrono::hours{ 24 };
    constexpr auto interval = std::chrono::seconds{ 10s };

    auto intervals = bitbase_client.get_intervals(symbol, exchange, timestamp_start, timestamp_end, interval);
    */

    //auto data_loader = torch::data::

    auto dataset = TradeDataset{};

    auto model = RepresentationLearner{};

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(dataset.size().value()));

    auto optimizer = torch::optim::Adam{ model->parameters(), torch::optim::AdamOptions(1e-3) };

    auto n_epochs = 10;

    for (auto epoch = 1; epoch <= n_epochs; ++epoch) {


        for (auto& batch : *data_loader) {

            model->zero_grad();

            auto observations = torch::ones({ batch_size, 1, n_observations, segment_length });
            auto future_positive = torch::ones({ batch_size, 1, n_predictions, segment_length });
            auto future_negative = torch::ones({ batch_size, 1, n_predictions * negative_count, segment_length });
            auto [accuracy, info_nce] = model->forward(observations, future_positive, future_negative);

            std::cout << accuracy << ", " << info_nce << std::endl;

            //logger.info("main data_loader batch size: %d", batch.size());
            //std::cout << std::endl;
        }


    }
    
}

/*
class TimeDistributed(nn.Module):
    def __init__(self, layer, time_steps, *args):
        super(TimeDistributed, self).__init__()

        self.layers = nn.ModuleList([layer(*args) for i in range(time_steps)])

    def forward(self, x):

        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([])
        for i in range(time_steps):
          output_t = self.layers[i](x[:, i, :, :, :])
          output_t  = y.unsqueeze(1)
          output = torch.cat((output, output_t ), 1)
        return output

x = torch.rand(20, 100, 1, 5, 9)
model = TimeDistributed(nn.Conv2d, time_steps = 100, 1, 8, (3, 3) , 2,   1 ,True)
output = model(x)
*/


/*
#include <torch/torch.h>

#include "custom_dataset.h"
#include "model.h"

int main()
{
    // Load the model.
    ConvNet model(3, 64, 64); // CHW

    // Generate your data set. At this point you can add transforms to you data set, e.g. stack your
    // batches into a single tensor.
    std::string file_names_csv = "../file_names.csv";
    auto data_set = CustomDataset(file_names_csv).map(torch::data::transforms::Stack<>());

    // Generate a data loader.
    int64_t batch_size = 32;
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        data_set,
        batch_size);

    // Chose and optimizer.
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(1e-3));

    // Train the network.
    int64_t n_epochs = 10;
    int64_t log_interval = 10;
    int dataset_size = data_set.size().value();

    // Record best loss.
    float best_mse = std::numeric_limits<float>::max();

    for (int epoch = 1; epoch <= n_epochs; epoch++) {

        // Track loss.
        size_t batch_idx = 0;
        float mse = 0.; // mean squared error
        int count = 0;

        for (auto& batch : *data_loader) {
            auto imgs = batch.data;
            auto labels = batch.target.squeeze();

            imgs = imgs.to(torch::kF32);
            labels = labels.to(torch::kInt64);

            optimizer.zero_grad();
            auto output = model(imgs);
            auto loss = torch::nll_loss(output, labels);

            loss.backward();
            optimizer.step();

            mse += loss.template item<float>();

            batch_idx++;
            if (batch_idx % log_interval == 0)
            {
                std::printf(
                    "\rTrain Epoch: %d/%ld [%5ld/%5d] Loss: %.4f",
                    epoch,
                    n_epochs,
                    batch_idx * batch.data.size(0),
                    dataset_size,
                    loss.template item<float>());
            }

            count++;
        }

        mse /= (float)count;
        printf(" Mean squared error: %f\n", mse);

        if (mse < best_mse)
        {
            torch::save(model, "../best_model.pt");
            best_mse = mse;
        }
    }

    return 0;
}
*/