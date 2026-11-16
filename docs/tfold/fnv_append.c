// fnv_append.c: compute FNV-1a-64 over every byte of a file, then append the
// 8-byte little-endian trailer. Matches src/core/tfold_file.zig's fnv1a64_scalar.
// Build: gcc -O3 -o fnv_append fnv_append.c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s path\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 2; }
    uint64_t h = 0xcbf29ce484222325ULL;   // FNV offset basis
    const uint64_t P = 0x100000001b3ULL;  // FNV prime
    unsigned char buf[1 << 20];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        for (size_t i = 0; i < n; i++) { h ^= buf[i]; h *= P; }
    }
    fclose(f);
    FILE *fo = fopen(argv[1], "ab");
    if (!fo) { perror("fopen ab"); return 3; }
    for (int i = 0; i < 8; i++) {
        unsigned char b = (h >> (i * 8)) & 0xff;
        if (fwrite(&b, 1, 1, fo) != 1) { perror("fwrite"); return 4; }
    }
    fclose(fo);
    printf("fnv=%016lx appended to %s\n", h, argv[1]);
    return 0;
}
