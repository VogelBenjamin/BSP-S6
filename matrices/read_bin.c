#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *f = fopen("./matrix_data/binary_mat.bin", "rb");
    int M, N, nz;
    fread(&M, sizeof(int), 1, f);
    fread(&N, sizeof(int), 1, f);
    fread(&nz, sizeof(int), 1, f);

    for (int i = 0; i < nz; i++) {
        int I, J;
        double val;
        fread(&I, sizeof(int), 1, f);
        fread(&J, sizeof(int), 1, f);
        fread(&val, sizeof(double), 1, f);
        printf("Entry %d: (%d, %d) = %lf\n", i, I, J, val);
    }
    fclose(f);
}
