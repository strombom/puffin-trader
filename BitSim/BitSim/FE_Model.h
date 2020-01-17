#pragma once

#include "BitSim.h"


#pragma warning(push, 0)
#pragma warning(disable: 4146)
#include <torch/torch.h>
#pragma warning(pop)


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
    TimeDistributedImpl(torch::nn::Module module, const int time_steps)
    {
        for (int idx = 0; idx < time_steps; ++idx) {
            layers->push_back(module);
        }
    }

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::ModuleList layers;
};
TORCH_MODULE(TimeDistributed);


struct FeatureEncoderImpl : public torch::nn::Module
{
    FeatureEncoderImpl(const std::string &name = "feature_encoder_cnn") :
        encoder(register_module(name, torch::nn::Sequential{
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::n_channels, BitSim::feature_size, 10}.stride(5).padding(3).with_bias(false)},
            torch::nn::BatchNorm{BitSim::feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::feature_size, BitSim::feature_size, 8}.stride(4).padding(2).with_bias(false)},
            torch::nn::BatchNorm{BitSim::feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::feature_size, BitSim::feature_size, 4}.stride(2).padding(1).with_bias(false)},
            torch::nn::BatchNorm{BitSim::feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::feature_size, BitSim::feature_size, 4}.stride(2).padding(1).with_bias(false)},
            torch::nn::BatchNorm{BitSim::feature_size},
            torch::nn::Functional{torch::relu},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::feature_size, BitSim::feature_size, 4}.stride(2).padding(1).with_bias(false)},
            torch::nn::BatchNorm{BitSim::feature_size},
            torch::nn::Functional{torch::relu}
            })) {}
    
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential encoder;
};
TORCH_MODULE(FeatureEncoder);


struct FeaturePredictorImpl : public torch::nn::Module
{
    FeaturePredictorImpl(const std::string &name = "feature_preditor_gru") :
        gru(register_module(name,
            torch::nn::GRU{
                torch::nn::GRUOptions{ BitSim::feature_size, BitSim::feature_size }
                    .layers(1)
                    .bidirectional(false)
                    .batch_first(true)
            }
        )) {}

    torch::Tensor forward(torch::Tensor observed_features);

private:
    torch::nn::GRU gru;
};
TORCH_MODULE(FeaturePredictor);


struct RepresentationLearnerImpl : public torch::nn::Module
{
    RepresentationLearnerImpl(const std::string& encoder_name = "feature encoder", 
                              const std::string& predictor_name = "feature predictor") :
        feature_encoder(register_module(encoder_name, FeatureEncoder{})),
        feature_predictor(register_module(predictor_name, FeaturePredictor{})) {}

    std::tuple<double, double> forward_fit(
        torch::Tensor past_observations,   // BxCxNxL (2x1x4x160)
        torch::Tensor future_positives,    // BxCxNxL (2x1x1x160)
        torch::Tensor future_negatives);   // BxCxNxL (2x1x9x160)

    void forward_predict(void);

private:
    FeatureEncoder feature_encoder;
    FeaturePredictor feature_predictor;
};
TORCH_MODULE(RepresentationLearner);
