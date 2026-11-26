#ifndef SD_RUNTIME_MINIMAX_H3_IMAGE_SCHEDULE_H
#define SD_RUNTIME_MINIMAX_H3_IMAGE_SCHEDULE_H

#include <array>

namespace MiniMaxH3 {
    inline constexpr std::array<float, 7> pdd_sigmas = {
        1.f, 0.9675236940383911f, 0.8875485062599182f, 0.75f,
        0.5327739119529724f, 0.23200887441635132f, 0.f};
}

#endif
