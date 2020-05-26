#pragma once
#include "pch.h"

#include "BitBotConstants.h"


struct FeatureEncoderImpl : public torch::nn::Module
{
    FeatureEncoderImpl(const std::string& name = "feature_encoder_cnn") :
        encoder(register_module(name, torch::nn::Sequential{
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::n_channels, BitSim::feature_size, 4}.stride(2).padding(1).bias(false)},
            torch::nn::BatchNorm1d{BitSim::feature_size},
            torch::nn::ReLU6{},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::feature_size, BitSim::feature_size, 4}.stride(2).padding(1).bias(false)},
            torch::nn::BatchNorm1d{BitSim::feature_size},
            torch::nn::ReLU6{},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::feature_size, BitSim::feature_size, 4}.stride(2).padding(1).bias(false)},
            torch::nn::BatchNorm1d{BitSim::feature_size},
            torch::nn::ReLU6{},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::feature_size, BitSim::feature_size, 4}.stride(2).padding(1).bias(false)},
            torch::nn::BatchNorm1d{BitSim::feature_size},
            torch::nn::ReLU6{},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::feature_size, BitSim::feature_size, 4}.stride(2).padding(1).bias(false)},
            torch::nn::BatchNorm1d{BitSim::feature_size},
            torch::nn::ReLU6{},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::feature_size, BitSim::feature_size, 4}.stride(2).padding(1).bias(false)},
            torch::nn::BatchNorm1d{BitSim::feature_size},
            torch::nn::ReLU6{},
            torch::nn::Conv1d{torch::nn::Conv1dOptions{BitSim::feature_size, BitSim::feature_size, 4}.stride(2).padding(1).bias(false)},
            torch::nn::BatchNorm1d{BitSim::feature_size},
            torch::nn::Sigmoid{}
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
                    .num_layers(1)
                    .bidirectional(false)
                    .batch_first(true)
            }
        )),
        sigmoid(register_module(name + "_sigmoid", torch::nn::Sigmoid{})) {
    
        //torch::nn::init
    }

    torch::Tensor forward(torch::Tensor observed_features);

private:
    torch::nn::GRU gru;
    torch::nn::Sigmoid sigmoid;
};
TORCH_MODULE(FeaturePredictor);


struct RepresentationLearnerImpl : public torch::nn::Module
{
    RepresentationLearnerImpl(const std::string& encoder_name = "feature encoder", 
                              const std::string& predictor_name = "feature predictor") :
        feature_encoder(register_module(encoder_name, FeatureEncoder{})),
        feature_predictor(register_module(predictor_name, FeaturePredictor{})) {}

    std::tuple<torch::Tensor, double> forward_fit(
        torch::Tensor past_observations,   // BxCxNxL (2x1x4x160)
        torch::Tensor future_positives,    // BxCxNxL (2x1x1x160)
        torch::Tensor future_negatives);   // BxCxNxL (2x1x9x160)

    torch::Tensor forward_predict(torch::Tensor observations);

private:
    FeatureEncoder feature_encoder;
    FeaturePredictor feature_predictor;
};
TORCH_MODULE(RepresentationLearner);
