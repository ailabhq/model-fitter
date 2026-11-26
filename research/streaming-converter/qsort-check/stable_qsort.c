// Interposes libc qsort with a stable merge sort (the algorithm glibc 2.36's qsort uses when it can allocate).
#include <stdlib.h>
#include <string.h>
static void msort(char *b, size_t n, size_t s, int (*c)(const void *, const void *), char *t) {
    if (n < 2) return;
    size_t h = n / 2, i = 0, j = h, k = 0;
    msort(b, h, s, c, t);
    msort(b + h * s, n - h, s, c, t);
    while (i < h && j < n) {
        if (c(b + i * s, b + j * s) <= 0) memcpy(t + k++ * s, b + i++ * s, s);
        else memcpy(t + k++ * s, b + j++ * s, s);
    }
    while (i < h) memcpy(t + k++ * s, b + i++ * s, s);
    while (j < n) memcpy(t + k++ * s, b + j++ * s, s);
    memcpy(b, t, n * s);
}
static void stable_qsort(void *b, size_t n, size_t s, int (*c)(const void *, const void *)) {
    char *t = malloc(n * s);
    msort(b, n, s, c, t);
    free(t);
}
__attribute__((used)) static const struct { const void *r, *o; } interposers[] __attribute__((section("__DATA,__interpose"))) = {
    {(const void *)stable_qsort, (const void *)qsort}};
