#include "pch.h"

#include "FE_Inference.h"


FE_Inference::FE_Inference(const std::string& file_path)
{
    auto archive = torch::serialize::InputArchive{};
    archive.load_from(file_path);
    model->load(archive);
}

