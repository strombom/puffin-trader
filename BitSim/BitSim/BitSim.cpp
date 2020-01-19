
#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/torch.h>
#pragma warning(pop)

#include "BitBaseClient.h"
#include "Logger.h"
#include "FE_DataLoader.h"
#include "FE_Model.h"

#include <iostream>




int main() {
    logger.info("BitSim started");

    auto bitbase_client = BitBaseClient();

    constexpr auto symbol = "XBTUSD";
    constexpr auto exchange = "BITMEX";
    constexpr auto timestamp_start = date::sys_days(date::year{ 2019 } / 06 / 01) + std::chrono::hours{ 0 } + std::chrono::minutes{ 0 };
    constexpr auto timestamp_end = date::sys_days(date::year{ 2019 } / 07 / 01) + std::chrono::hours{ 0 };
    constexpr auto interval = std::chrono::seconds{ 10s };

    auto intervals = bitbase_client.get_intervals(symbol, exchange, timestamp_start, timestamp_end, interval);
    
    auto dataset = TradeDataset{ std::move(intervals) };
    
    auto model = RepresentationLearner{};
    model->to(c10::DeviceType::CUDA);

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(BitSim::batch_size).workers(10));

    auto optimizer = torch::optim::Adam{ model->parameters(), torch::optim::AdamOptions(1e-3) };
    
    for (auto epoch = 1; epoch <= BitSim::n_epochs; ++epoch) {

        for (auto& batch : *data_loader) {

            optimizer.zero_grad();

            auto past_observations = batch.past_observations;
            auto future_positives  = batch.future_positives; // torch::ones({ BitSim::batch_size, 1, BitSim::n_positive,     BitSim::observation_length });
            auto future_negatives  = batch.future_negatives; // torch::ones({ BitSim::batch_size, 1, BitSim::n_negative,     BitSim::observation_length });

            //std::cout << "past_observations: " << past_observations.sizes() << std::endl;
            //std::cout << "future_positives: " << future_positives.sizes() << std::endl;
            //std::cout << "future_negatives: " << future_negatives.sizes() << std::endl;

            auto [info_nce_loss, accuracy] = model->forward_fit(past_observations, future_positives, future_negatives);
            //std::cout << accuracy << ", " << info_nce << std::endl;

            info_nce_loss.backward();
            optimizer.step();


            logger.info("main data_loader loss: %f", info_nce_loss.item().to<double>()); // batch.size());
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

    //auto a = std::array<float, 6>{ 1.0f, 2.0f, 3.0f, 1.1f, 1.2f, 1.3f };
    //auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    //torch::Tensor t = torch::from_blob(t.data(), { 3 }, opts).to(torch::kInt64);
    //auto t = torch::from_blob(a.data(), { 2, 3 }, opts).clone();
    //std::cout << t << std::endl;

    std::array<float, BitSim::observation_length> price;
    std::array<float, BitSim::observation_length> vol_buy;
    std::array<float, BitSim::observation_length> vol_sell;

    torch::Tensor past_observations,   // BxCxNxL (2x1x4x160)
    torch::Tensor future_positives,    // BxCxNxL (2x1x1x160)
    torch::Tensor future_negatives)    // BxCxNxL (2x1x9x160)

    auto past_observations = torch::ones({ BitSim::batch_size, 1, BitSim::n_observations, BitSim::observation_length });
    auto future_positives = torch::ones({ BitSim::batch_size, 1, BitSim::n_positive,     BitSim::observation_length });
    auto future_negatives = torch::ones({ BitSim::batch_size, 1, BitSim::n_negative,     BitSim::observation_length });


Seq2seq https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
https://pytorch.org/cppdocs/frontend.html
https://pytorch.org/tutorials/advanced/cpp_frontend.html
https://discuss.pytorch.org/t/usage-of-cross-entropy-loss/14841/2
https://pytorch.org/docs/stable/nn.html#gru
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


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