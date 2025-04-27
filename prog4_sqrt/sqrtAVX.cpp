#include <immintrin.h>

extern void sqrtSerial(int N, float startGuess, float* values, float* output);

void sqrtAVX(
    int N, 
    float initialGuess,
    float values[],
    float output[])
{
    __m256 x;
    __m256 error;
    __m256 guess;
    __m256 maskNew;
    __m256 errorNew;
    __m256 guessNew;

    __m256 one = _mm256_set1_ps(1.f);
    __m256 three = _mm256_set1_ps(3.f);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 maskSign = _mm256_set1_ps(-0.f);
    __m256 kThreshold = _mm256_set1_ps(0.00001f);

    int i = 0;
    for (; i < N; i += 8) {
        x = _mm256_loadu_ps(values + i);
        guess = _mm256_set1_ps(initialGuess);
        error = _mm256_andnot_ps(
            maskSign,
            _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guess, guess), x), one)
        );

        maskNew = _mm256_cmp_ps(error, kThreshold, _CMP_GT_OS);
        while (_mm256_movemask_ps(maskNew) != 0) {
            guessNew = _mm256_mul_ps(
                _mm256_sub_ps(
                    _mm256_mul_ps(guess, three), 
                    _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(x, guess), guess), guess)
                ), 
                half
            );
            guess = _mm256_blendv_ps(guess, guessNew, maskNew);
            errorNew = _mm256_andnot_ps(
                maskSign,
                _mm256_sub_ps(_mm256_mul_ps(_mm256_mul_ps(guess, guess), x), one)
            );
            error = _mm256_blendv_ps(error, errorNew, maskNew);
            maskNew = _mm256_cmp_ps(error, kThreshold, _CMP_GT_OS);
        }

        _mm256_storeu_ps(output + i, _mm256_mul_ps(x, guess));
    }

    i -= 8;
    sqrtSerial(N - i, initialGuess, values + i, output + i);
}

