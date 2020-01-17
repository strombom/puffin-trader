
#include "FE_Model.h"


torch::Tensor TimeDistributedImpl::forward(torch::Tensor x) {
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


torch::Tensor FeatureEncoderImpl::forward(torch::Tensor x) {
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


torch::Tensor FeaturePredictorImpl::forward(torch::Tensor observed_features) {
    // TODO: Attention? https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
    const auto initial_hidden = torch::zeros({ 1, BitSim::batch_size, BitSim::feature_size });
    auto gru_result = gru->forward(observed_features, initial_hidden);
    auto prediction = gru_result.output;                       // BxNxC
    prediction = prediction.select(1, prediction.size(1) - 1); // BxC
    return prediction;
};


std::tuple<double, double> RepresentationLearnerImpl::forward_fit(
    torch::Tensor past_observations,   // BxCxNxL (2x1x4x160)
    torch::Tensor future_positives,    // BxCxNxL (2x1x1x160)
    torch::Tensor future_negatives)    // BxCxNxL (2x1x9x160)
{
    auto past_features = feature_encoder->forward(past_observations); // BxCxN (2x512x4)
    auto positive_features = feature_encoder->forward(future_positives);  // BxCxN (2x512x1)
    auto negative_features = feature_encoder->forward(future_negatives);  // BxCxN (2x512x9)
    std::cout << "past_features: " << past_features.sizes() << std::endl;
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

void RepresentationLearnerImpl::forward_predict(void) {

}