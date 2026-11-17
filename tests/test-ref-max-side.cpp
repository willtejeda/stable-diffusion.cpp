#undef NDEBUG
#include "common/common.h"
#include "runtime/image_preprocess.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static SDImageOwner image(unsigned w, unsigned h, unsigned channels) {
    auto* pixels = static_cast<uint8_t*>(std::malloc(size_t(w) * h * channels));
    assert(pixels);
    for (size_t i = 0; i < size_t(w) * h * channels; ++i) pixels[i] = static_cast<uint8_t>(i % 251);
    return SDImageOwner({w, h, channels, pixels});
}

int main() {
    SDGenerationParams p;
    assert(p.ref_max_side == 0);
    for (const char* bad : {"-1", "8193", "1.0", "true", "null", "\"768\"", "999999999999999999999"}) {
        assert(!p.from_json_str(std::string("{\"ref_max_side\":") + bad + "}"));
    }
    assert(p.from_json_str("{\"ref_max_side\":8192}"));
    assert(p.from_json_str("{\"ref_max_side\":768}"));
    p.width = 1152;
    p.height = 640;
    p.ref_image_args = "resize_before_vae=true";
    p.ref_images.push_back(image(1200, 600, 4));
    p.ref_images.push_back(image(400, 1000, 1));
    p.ref_images.push_back(image(300, 200, 3));
    p.ref_images.push_back(image(768, 768, 4));
    SDGenerationParams copy = p;
    const auto* original = p.ref_images[0].get().data;
    std::vector<uint8_t> bytes(original, original + 1200 * 600 * 4);
    assert(p.resolve("", "", false, false));
    auto view = p.to_sd_img_gen_params_t();
    sd::ImagePreprocessor prep;
    assert(prep.prepare_inputs(view, view.width, view.height));
    assert(view.width == 1152 && view.height == 640);
    assert(view.ref_images_count == 4);
    assert(view.ref_images[0].width == 768 && view.ref_images[0].height == 384 && view.ref_images[0].channel == 4);
    assert(view.ref_images[1].width == 307 && view.ref_images[1].height == 768 && view.ref_images[1].channel == 1);
    assert(view.ref_images[2].width == 300 && view.ref_images[2].height == 200);
    assert(view.ref_images[3].width == 768 && view.ref_images[3].height == 768);
    assert(view.ref_max_side == 768);
    assert(!sd::ref_resize_before_vae(true, view));
    assert(!sd::ref_resize_before_vae(false, view));
    assert(std::equal(bytes.begin(), bytes.end(), original));
    assert(std::equal(bytes.begin(), bytes.end(), copy.ref_images[0].get().data));
    assert(p.resolve("", "", false, false));
    auto repeated = p.to_sd_img_gen_params_t();
    sd::ImagePreprocessor prep2;
    assert(prep2.prepare_inputs(repeated, repeated.width, repeated.height));
    assert(repeated.ref_images[0].width == 768 && p.ref_images[0].get().width == 1200);
    assert(std::equal(bytes.begin(), bytes.end(), original));
    p.ref_max_side = 0;
    view = p.to_sd_img_gen_params_t();
    sd::ImagePreprocessor legacy;
    assert(legacy.prepare_inputs(view, view.width, view.height));
    assert(view.ref_images[0].width == 1200 && view.ref_images[0].height == 600);
    assert(std::string(view.ref_image_args) == "resize_before_vae=true");
    assert(sd::ref_resize_before_vae(true, view));
    assert(!sd::ref_resize_before_vae(false, view));
    p.ref_max_side = 768;
    p.image_preprocess = "target=ref,index=2,mode=stretch,width=1600,height=800,filter=bilinear";
    auto transformed = p.to_sd_img_gen_params_t();
    sd::ImagePreprocessor transformed_prep(transformed.image_preprocess.rules);
    assert(transformed_prep.prepare_inputs(transformed, transformed.width, transformed.height));
    assert(transformed.ref_images[2].width == 768 && transformed.ref_images[2].height == 384);
    // Two reference descriptors may share caller-owned pixels; neither may be
    // modified and each output must own a separate transformed buffer.
    std::vector<uint8_t> shared(1000 * 500 * 4);
    for (size_t i = 0; i < shared.size(); i += 4) {
        shared[i] = 80;
        shared[i + 1] = 120;
        shared[i + 2] = 160;
        shared[i + 3] = 255;
    }
    const auto shared_before = shared;
    sd_image_t refs[] = {{1000, 500, 4, shared.data()}, {1000, 500, 4, shared.data()}};
    sd_img_gen_params_t raw;
    sd_img_gen_params_init(&raw);
    raw.ref_images = refs;
    raw.ref_images_count = 2;
    raw.ref_max_side = 500;
    sd::ImagePreprocessor shared_prep;
    assert(shared_prep.prepare_inputs(raw, 512, 512));
    assert(refs[0].data == shared.data() && refs[1].data == shared.data());
    assert(shared == shared_before);
    assert(raw.ref_images[0].data != shared.data());
    assert(raw.ref_images[0].data != raw.ref_images[1].data);
    assert(raw.ref_images[0].width == 500 && raw.ref_images[0].height == 250);
    for (int c = 0; c < 4; ++c)
        assert(std::abs(int(raw.ref_images[0].data[c]) - int(shared[c])) <= 1);
    std::vector<uint8_t> gray(1000 * 500, 91);
    sd_image_t gray_ref = {1000, 500, 1, gray.data()};
    sd_img_gen_params_t gray_params;
    sd_img_gen_params_init(&gray_params);
    gray_params.ref_images = &gray_ref;
    gray_params.ref_images_count = 1;
    gray_params.ref_max_side = 500;
    sd::ImagePreprocessor gray_prep;
    assert(gray_prep.prepare_inputs(gray_params, 512, 512));
    assert(gray_params.ref_images[0].channel == 1);
    assert(std::abs(int(gray_params.ref_images[0].data[0]) - 91) <= 1);
    assert(std::all_of(gray.begin(), gray.end(), [](uint8_t v) { return v == 91; }));
    // Qwen requires multiples of 16; padding must not alter content or caller buffers.
    {
        auto odd = image(768, 385, 4);
        auto small = image(300, 201, 1);
        auto odd_original = odd.get();
        auto small_original = small.get();
        std::vector<uint8_t> before(odd_original.data, odd_original.data + 768 * 385 * 4);
        sd_image_t input[] = {odd_original, small_original};
        sd_img_gen_params_t qwen;
        sd_img_gen_params_init(&qwen);
        qwen.ref_images = input;
        qwen.ref_images_count = 2;
        qwen.ref_max_side = 768;
        qwen.width = 1152;
        qwen.height = 640;
        sd::ImagePreprocessor qwen_prep;
        assert(qwen_prep.prepare_inputs(qwen, qwen.width, qwen.height, true));
        assert(qwen_prep.prepare_qwen_references(qwen));
        assert(qwen.width == 1152 && qwen.height == 640);
        assert(qwen.ref_images[0].width == 768 && qwen.ref_images[0].height == 400);
        assert(qwen.ref_images[1].width == 304 && qwen.ref_images[1].height == 208);
        assert(qwen.ref_images[1].channel == 1);
        assert(std::equal(before.begin(), before.end(), odd_original.data));
        assert(std::equal(before.begin(), before.end(), qwen.ref_images[0].data));
        for (unsigned y = 0; y < 201; ++y)
            assert(std::equal(small_original.data + y * 300, small_original.data + (y + 1) * 300,
                              qwen.ref_images[1].data + y * 304));
    }
    {
        auto source = image(768, 385, 3);
        sd_image_t ref = source.get();
        sd_img_gen_params_t qwen;
        sd_img_gen_params_init(&qwen);
        qwen.ref_images = &ref;
        qwen.ref_images_count = 1;
        qwen.ref_max_side = 767;
        sd::ImagePreprocessor qwen_prep;
        assert(qwen_prep.prepare_inputs(qwen, 512, 512, true));
        assert(qwen_prep.prepare_qwen_references(qwen));
        assert(qwen.ref_images[0].width == 752 && qwen.ref_images[0].height == 384);
        assert(ref.width == 768 && ref.height == 385);
    }
    {
        auto source = image(30, 20, 3);
        sd_image_t ref = source.get();
        sd_img_gen_params_t qwen;
        sd_img_gen_params_init(&qwen);
        qwen.ref_images = &ref;
        qwen.ref_images_count = 1;
        qwen.ref_max_side = 1;
        sd::ImagePreprocessor qwen_prep;
        assert(!qwen_prep.prepare_inputs(qwen, 512, 512, true));
    }
    {
        auto source = image(300, 201, 3);
        sd_image_t ref = source.get();
        sd_img_gen_params_t qwen;
        sd_img_gen_params_init(&qwen);
        qwen.ref_images = &ref;
        qwen.ref_images_count = 1;
        sd::ImagePreprocessor qwen_prep;
        assert(qwen_prep.prepare_inputs(qwen, 512, 512, true));
        assert(qwen_prep.prepare_qwen_references(qwen));
        assert(qwen.ref_images[0].width == 300 && qwen.ref_images[0].height == 201);
    }
    SDGenerationParams empty;
    empty.ref_max_side = 768;
    view = empty.to_sd_img_gen_params_t();
    assert(view.ref_images_count == 0 && std::string(view.ref_image_args).empty());
    assert(sd::ref_resize_before_vae(true, view));
    for (const char* value : {"-1", "8193", "1.0", "768junk", "999999999999999999999"}) {
        SDGenerationParams cli;
        const char* args[] = {"test", "--ref-max-side", value};
        assert(!parse_options(3, args, {cli.get_options()}));
    }
    SDGenerationParams cli;
    const char* args[] = {"test", "--ref-max-side", "768"};
    assert(parse_options(3, args, {cli.get_options()}));
    assert(cli.ref_max_side == 768);
}
