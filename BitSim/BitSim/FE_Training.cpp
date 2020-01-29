#include "FE_Training.h"

#include "FE_DataLoader.h"
#include "FE_Scheduler.h"
#include "FE_Model.h"
#include "DateTime.h"
#include "Logger.h"
#include "Utils.h"

#include <iostream>
#include <fstream>

#pragma warning(push, 0)
#pragma warning(disable: 4146)
//#include <torch/torch.h>
#pragma warning(pop)


void FE_Training::test_learning_rate(void)
{


}

void FE_Training::measure_observations(void)
{
    auto dataset = TradeDataset{ std::move(intervals) };
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions{}.batch_size(BitSim::batch_size).workers(10));

    auto idx = 0;
    for (auto& batch : *data_loader) {

        Utils::save_tensor(batch.past_observations, std::string{ "past_observations_" } + std::to_string(idx) + std::string{ ".tensor" });
        Utils::save_tensor(batch.future_positives,  std::string{ "future_positives_" }  + std::to_string(idx) + std::string{ ".tensor" });
        Utils::save_tensor(batch.future_negatives,  std::string{ "future_negatives_" }  + std::to_string(idx) + std::string{ ".tensor" });
        ++idx;
    }
}

void FE_Training::train(void)
{
    const auto lr_test = false;

    auto timer = Timer();

    auto dataset = TradeDataset{ std::move(intervals) };

    auto model = RepresentationLearner{};
    model->to(c10::DeviceType::CUDA);

    timer.restart();
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset),
        torch::data::DataLoaderOptions{}.batch_size(BitSim::batch_size).workers(10));
    timer.print_elapsed("DataLoader");

    auto optimizer = torch::optim::SGD{ model->parameters(), torch::optim::SGDOptions{0.01}.momentum(0.9) };

    const auto start_iteration = 0;    
    auto scheduler = uptrFE_Scheduler{};
    if (lr_test) {
        scheduler = std::make_unique<FE_Scheduler>(BitSim::n_batches, 0.0000001, 2.0, 1.0, 1.0, start_iteration, true);
    }
    else {
        scheduler = std::make_unique<FE_Scheduler>(BitSim::n_batches, 0.01, 0.001, 0.98, 0.90, start_iteration, false);
    }

    timer.restart();
    for (auto& batch : *data_loader) {
        //timer.print_elapsed("Batch loaded");
        timer.restart();

        optimizer.zero_grad();

        auto past_observations = batch.past_observations;
        auto future_positives = batch.future_positives;
        auto future_negatives = batch.future_negatives;

        auto [info_nce_loss, accuracy] = model->forward_fit(past_observations, future_positives, future_negatives);

        info_nce_loss.backward();
        optimizer.step();
        //timer.print_elapsed("Step");

        const auto [learning_rate, momentum] = scheduler->calc(info_nce_loss.item().to<double>());
        optimizer.options.learning_rate(learning_rate);
        optimizer.options.momentum(momentum);
        logger.info("step loss(%f) lr(%f) mom(%f)", info_nce_loss.item().to<double>(), learning_rate, momentum);
        
        if (scheduler->finished()) {
            break;
        }
        timer.restart();
    }
}


//file.close();
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

