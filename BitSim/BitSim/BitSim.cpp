
#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/torch.h>
#pragma warning(pop)

#include "BitBaseClient.h"
#include "Logger.h"

#include <iostream>

namespace params {
    constexpr auto batch_size = 2;
    constexpr auto segment_length = 160;
    constexpr auto n_observations = 4;
    constexpr auto n_predictions = 2;
    constexpr auto n_positive = 1;
    constexpr auto n_negative = 3;

    constexpr auto feature_size = 256;
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


struct TimeDistributedImpl : public torch::nn::Module
{
    TimeDistributedImpl(torch::nn::Module module, int time_steps)
    {
        for (int idx = 0; idx < time_steps; ++idx) {
            layers->push_back(module);
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        auto output = torch::Tensor{};
        for (auto layer : *layers) {
            //auto output_t = layer->forward();
        }
        // x                                                            // BxCxNxL  (2x1x4x160)
        //x = x.reshape({ x.size(0), x.size(1), x.size(2) * x.size(3) }); // BxCx(NL) (2x1x640)
        //x = encoder->forward(x);                                        // BxCxN    (2x512x4)
        //x = x.transpose(1, 2);                                          // BxNxC    (2x4x512)
        return x;
    };

private:
    torch::nn::ModuleList layers;
};
TORCH_MODULE(TimeDistributed);

struct FeatureEncoderImpl : public torch::nn::Module
{
    FeatureEncoderImpl(void) {
        encoder = register_module("feature_encoder_cnn", torch::nn::Sequential{
            torch::nn::Conv1d{torch::nn::Conv1dOptions{1, params::feature_size, 10}.stride(5).padding(3).with_bias(false)},
            torch::nn::BatchNorm{params::feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{params::feature_size, params::feature_size, 8}.stride(4).padding(2).with_bias(false)},
            torch::nn::BatchNorm{params::feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{params::feature_size, params::feature_size, 4}.stride(2).padding(1).with_bias(false)},
            torch::nn::BatchNorm{params::feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{params::feature_size, params::feature_size, 4}.stride(2).padding(1).with_bias(false)},
            torch::nn::BatchNorm{params::feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{params::feature_size, params::feature_size, 4}.stride(2).padding(1).with_bias(false)},
            torch::nn::BatchNorm{params::feature_size},
            torch::nn::Functional{torch::relu}
            });
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = encoder->forward(x);
        return x;

        //for (auto encoder : encoders) {

        //}
        // x                                                              // BxCxNxL  (2x1x4x160)
        //x = x.reshape({ x.size(0), x.size(1), x.size(2) * x.size(3) }); // BxCx(NL) (2x1x640)
        //x = encoder->forward(x);                                        // BxCxN    (2x512x4)
        //x = x.transpose(1, 2);                                          // BxNxC    (2x4x512)
        //return x;
    };

private:
    torch::nn::Sequential encoder;
};
TORCH_MODULE(FeatureEncoder);


struct FeaturePredictorImpl : public torch::nn::Module
{
    // TODO: Attention? https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

    FeaturePredictorImpl(void) {
        gru = register_module("feature_preditor_gru", 
            torch::nn::GRU{ 
                torch::nn::GRUOptions{ params::feature_size, params::feature_size }
                    .layers(1)
                    .bidirectional(false)
                    .batch_first(true)
            }
        );
    }

    torch::Tensor forward(torch::Tensor observed_features) {
        const auto initial_hidden = torch::zeros({ 1, params::batch_size, params::feature_size });
        auto gru_result = gru->forward(observed_features, initial_hidden);
        auto prediction = gru_result.output;                       // BxNxC
        prediction = prediction.select(1, prediction.size(1) - 1); // BxC
        return prediction;
    };

private:
    torch::nn::GRU gru{ nullptr };
};
TORCH_MODULE(FeaturePredictor);


struct RepresentationLearnerImpl : public torch::nn::Module
{
    RepresentationLearnerImpl(void) {
        const auto feature_count = params::n_observations + params::n_predictions * (params::n_positive + params::n_negative);

        register_module("feature encoder", feature_encoder);
        register_module("feature predictor", feature_predictor);
    };

    std::tuple<double, double> forward_fit(
        torch::Tensor past_observations,   // BxCxNxL (2x1x4x160)
        torch::Tensor future_positives,    // BxCxNxL (2x1x1x160)
        torch::Tensor future_negatives)    // BxCxNxL (2x1x9x160)
    {
        auto past_features     = feature_encoder->forward(past_observations); // BxCxN (2x512x4)
        auto positive_features = feature_encoder->forward(future_positives);  // BxCxN (2x512x1)
        auto negative_features = feature_encoder->forward(future_negatives);  // BxCxN (2x512x9)
        std::cout << "past_features: "     << past_features.sizes()     << std::endl;
        std::cout << "positive_features: " << positive_features.sizes() << std::endl;
        std::cout << "negative_features: " << negative_features.sizes() << std::endl;

        auto prediction = feature_predictor->forward(past_features);      // BxNxC
        std::cout << "prediction: " << prediction.sizes() << std::endl;



        auto accuracy = 1.0;
        auto info_nce = -0.5;
        /*
            # TODO - optimized back - prop with K.categorical_cross_entropy() ?
            z, z_hat = inputs
            # z.shape() = (B, neg + 1, T, pred_steps, dim_z)
            z_hat = K.expand_dims(z_hat, axis = 1)  # add pos / neg example axis
            # z_pred.shape() = (B, 1, T, pred_steps, dim_z)
            logits = K.sum(z * z_hat, axis = -1)  # dot product
            # logits.shape() = (B, neg + 1, T, pred_steps)
            log_ll = logits[:, 0, ...] - tf.math.reduce_logsumexp(logits, axis = 1)
            # log_ll.shape() = (B, T, pred_steps)
            loss = -K.mean(log_ll, axis = [1, 2])
            # calculate prediction accuracy
            acc = K.cast(K.equal(K.argmax(logits, axis = 1), 0), 'float32')
            acc = K.mean(acc, axis = [0, 1])
        */

        return std::make_tuple(accuracy, info_nce);
    };

    void forward_predict(void) {

    }

private:
    FeatureEncoder feature_encoder{};
    FeaturePredictor feature_predictor{};
};
TORCH_MODULE(RepresentationLearner);


struct Batch {
    torch::Tensor past_observations;   // BxCxNxL (2x1x4x160)
    torch::Tensor future_positives;    // BxCxNxL (2x1x1x160)
    torch::Tensor future_negatives;    // BxCxNxL (2x1x9x160)

    Batch(size_t batch_size) {

    }



    /*
    std::array<float, params::segment_length> price;
    std::array<float, params::segment_length> vol_buy;
    std::array<float, params::segment_length> vol_sell;

        torch::Tensor past_observations,   // BxCxNxL (2x1x4x160)
        torch::Tensor future_positives,    // BxCxNxL (2x1x1x160)
        torch::Tensor future_negatives)    // BxCxNxL (2x1x9x160)

    auto past_observations = torch::ones({ params::batch_size, 1, params::n_observations, params::segment_length });
    auto future_positives = torch::ones({ params::batch_size, 1, params::n_positive,     params::segment_length });
    auto future_negatives = torch::ones({ params::batch_size, 1, params::n_negative,     params::segment_length });
    */
};


class TradeDataset : public torch::data::BatchDataset<TradeDataset, Batch, c10::ArrayRef<size_t>>
{
public:
    TradeDataset(Intervals intervals) : intervals(intervals) {}

    Batch get_batch(c10::ArrayRef<size_t> request) {
        return Batch{ request.size() };
    }

    c10::optional<size_t> size() const {
        return 4;
    }

private:
    Intervals intervals;

    /*
    c10::optional<size_t> size(void) const {
        return 4;
    }

    SegmentsBatch get(size_t index) {
        logger.info("TradeDataset:get get it %d", index);

        auto a = SegmentsBatch{};


        //auto a = std::array<float, 6>{ 1.0f, 2.0f, 3.0f, 1.1f, 1.2f, 1.3f };
        //auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        //torch::Tensor t = torch::from_blob(t.data(), { 3 }, opts).to(torch::kInt64);
        //auto t = torch::from_blob(a.data(), { 2, 3 }, opts).clone();
        //std::cout << t << std::endl;



        return a;
    }

    */
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

    auto intervals = Intervals{ time_point_us{ date::sys_days(date::year{2019} / 06 / 01) }, std::chrono::seconds{10s} };
    
    auto dataset = TradeDataset{ intervals };

    auto model = RepresentationLearner{};

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(params::batch_size));

    auto optimizer = torch::optim::Adam{ model->parameters(), torch::optim::AdamOptions(1e-3) };

    auto n_epochs = 10;

    for (auto epoch = 1; epoch <= n_epochs; ++epoch) {

        for (auto& batch : *data_loader) {

            model->zero_grad();

            auto past_observations = torch::ones({ params::batch_size, 1, params::n_observations, params::segment_length });
            auto future_positives  = torch::ones({ params::batch_size, 1, params::n_positive,     params::segment_length });
            auto future_negatives  = torch::ones({ params::batch_size, 1, params::n_negative,     params::segment_length });
            //auto [accuracy, info_nce] = model->forward_fit(past_observations, future_positives, future_negatives);
            //std::cout << accuracy << ", " << info_nce << std::endl;

            logger.info("main data_loader batch size: %d", 0); // batch.size());
            //std::cout << std::endl;


            /*
                MSE
                squared_error = (y_predicted - y_actual) ** 2
                sum_squared_error = np.sum(squared_error)
                mse = sum_squared_error / y_actual.size
            */
        }

    }

}


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