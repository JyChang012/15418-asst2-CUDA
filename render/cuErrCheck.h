//
// Created by Jiayu Chang on 6/10/23.
//

#ifndef ASST2_CUERRCHECK_H

#define ASST2_CUERRCHECK_H
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,
                "CUDA Error: %s at %s:%d\n",
                cudaGetErrorString(code),
                file,
                line);

        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif


#endif //ASST2_CUERRCHECK_H
