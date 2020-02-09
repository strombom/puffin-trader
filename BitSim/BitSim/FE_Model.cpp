#include "pch.h"

#include "FE_Model.h"


torch::Tensor FeatureEncoderImpl::forward(torch::Tensor x)
{
    const auto size = (int) x.sizes()[2];

    auto features = std::vector<torch::Tensor>{};
    features.reserve(size);
    auto ins = x.chunk(size, 2);

    for (auto&& in : ins) {
        // in: BxCxNxL (2x3x4x160)
        auto feature = encoder->forward(in.squeeze(2)); // BxNxL (2x3x256)
        feature = feature.transpose(1, 2);
        features.push_back(feature);
    }

    auto y = torch::cat(features, 1); // BxNxL (2x4x256)
    return y;
};


torch::Tensor FeaturePredictorImpl::forward(torch::Tensor observed_features)
{
    // TODO: Attention? https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    const auto initial_hidden = torch::zeros({ 1, BitSim::batch_size, BitSim::feature_size }).cuda();
    auto gru_result = gru->forward(observed_features, initial_hidden);
    auto prediction = gru_result.output;                       // BxNxC
    prediction = prediction.select(1, prediction.size(1) - 1); // BxC
    prediction = prediction.reshape({ prediction.size(0), 1, prediction.size(1) }); // Bx1xC
    prediction = sigmoid(prediction);
    return prediction;
};

torch::Tensor RepresentationLearnerImpl::forward_predict(torch::Tensor observation)
{
    auto features = feature_encoder->forward(observation);
    return features;
}

std::tuple<torch::Tensor, double> RepresentationLearnerImpl::forward_fit(
    torch::Tensor past_observations,   // BxCxNxL (2x1x4x160)
    torch::Tensor future_positives,    // BxCxNxL (2x1x(1x1)x160)
    torch::Tensor future_negatives)    // BxCxNxL (2x1x(1x9)x160)
{
    auto past_features     = feature_encoder->forward(past_observations); // BxCxN (2x512x4)
    auto positive_features = feature_encoder->forward(future_positives);  // BxCxN (2x512x1)
    auto negative_features = feature_encoder->forward(future_negatives);  // BxCxN (2x512x9)
    //std::cout << "past_features: " << past_features.sizes() << std::endl;
    //std::cout << "positive_features: " << positive_features.sizes() << std::endl;
    //std::cout << "negative_features: " << negative_features.sizes() << std::endl;
    
    auto prediction = feature_predictor->forward(past_features);      // BxNxC
    //std::cout << "prediction: " << prediction.sizes() << std::endl;

    auto target = torch::cat({ positive_features, negative_features }, 1);
    //std::cout << "target: " << target.sizes() << std::endl;

    auto logits_all = torch::einsum("bij,bkj->bk", { prediction, target }); // Dot product, sum features
    //std::cout << "logits_all: " << logits_all.sizes() << std::endl;

    auto logits_positive = logits_all.select(1, 0);
    //std::cout << "logits_positive: " << logits_positive.sizes() << std::endl;

    auto logits_ratio = logits_positive - logits_all.logsumexp(1);
    //std::cout << "logits_ratio: " << logits_ratio.sizes() << std::endl;
    //std::cout << logits_ratio << std::endl;

    auto info_nce_loss = -logits_ratio.mean();
    //std::cout << "info_nce_loss: " << info_nce_loss << std::endl;

    auto accuracy = 1.0;

    //auto info_nce = -0.5;
    /*
        # TODO - optimized back-prop with K.categorical_cross_entropy()?
        z_targ, z_pred = inputs
        # z_targ.shape() = (B, neg+1, T, pred_steps, dim_z)
        z_pred = K.expand_dims(z_pred, axis=1)  # add pos/neg example axis
        # z_pred.shape() = (B, 1, T, pred_steps, dim_z)
        logits = K.sum(z_targ * z_pred, axis=-1)  # dot product
        # logits.shape() = (B, neg+1, T, pred_steps)
        log_ll = logits[:, 0, ...] - tf.math.reduce_logsumexp(logits, axis=1)
        # log_ll.shape() = (B, T, pred_steps)
        loss = -K.mean(log_ll, axis=[1, 2])
        # calculate prediction accuracy
        acc = K.cast(K.equal(K.argmax(logits, axis=1), 0), 'float32')
        acc = K.mean(acc, axis=[0, 1])

    */

    /*
        MSE
        squared_error = (y_predicted - y_actual) ** 2
        sum_squared_error = np.sum(squared_error)
        mse = sum_squared_error / y_actual.size
    */

    return std::make_tuple(info_nce_loss, accuracy);
};
