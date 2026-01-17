#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define EPSILON 1e-15
typedef struct {
    size_t rows;
    size_t cols;
    double **data;
} Matrix;
Matrix* create_matrix(size_t rows, size_t cols) {
    Matrix *m = malloc(sizeof(Matrix));
    if (!m) return NULL;
    m->rows = rows;
    m->cols = cols;
    m->data = malloc(rows * sizeof(double *));
    if (!m->data) {
        free(m);
        return NULL;
    }
    m->data[0] = calloc(rows * cols, sizeof(double));
    if (!m->data[0]) {
        free(m->data);
        free(m);
        return NULL;
    }
    for (size_t i = 1; i < rows; i++) {
        m->data[i] = m->data[0] + i * cols;
    }
    return m;
}
void free_matrix(Matrix *m) {
    if (m) {
        if (m->data) {
            free(m->data[0]);
            free(m->data);
        }
        free(m);
    }
}
int decompose_LU(const Matrix *A, Matrix *L, Matrix *U, int *P) {
    size_t n = A->rows;
    for (size_t i = 0; i < n; i++) {
        P[i] = i;
        for (size_t j = 0; j < n; j++) {
            U->data[i][j] = A->data[i][j];
            L->data[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (size_t k = 0; k < n; k++) {
        double max_val = 0.0;
        size_t pivot_row = k;
        for (size_t i = k; i < n; i++) {
            if (fabs(U->data[i][k]) > max_val) {
                max_val = fabs(U->data[i][k]);
                pivot_row = i;
            }
        }
        if (max_val < EPSILON) return -1;
        if (pivot_row != k) {
            int temp_p = P[k]; P[k] = P[pivot_row]; P[pivot_row] = temp_p;
            double *temp_u = U->data[k]; U->data[k] = U->data[pivot_row]; U->data[pivot_row] = temp_u;
            for (size_t j = 0; j < k; j++) {
                double temp_l = L->data[k][j];
                L->data[k][j] = L->data[pivot_row][j];
                L->data[pivot_row][j] = temp_l;
            }
        }
        for (size_t i = k + 1; i < n; i++) {
            L->data[i][k] = U->data[i][k] / U->data[k][k];
            for (size_t j = k; j < n; j++) {
                U->data[i][j] -= L->data[i][k] * U->data[k][j];
            }
        }
    }
    return 0;
}
void solve_LU(const Matrix *L, const Matrix *U, const int *P, const double *b, double *x, size_t n) {
    double y[n];
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < i; j++) {
            sum += L->data[i][j] * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    for (int i = (int)n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (size_t j = i + 1; j < n; j++) {
            sum += U->data[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / U->data[i][i];
    }
}
int main() {
    size_t n = 10;
    Matrix *A = create_matrix(n, n);
    Matrix *L = create_matrix(n, n);
    Matrix *U = create_matrix(n, n);
    int *P = malloc(n * sizeof(int));
    double *b = malloc(n * sizeof(double));
    double *x = malloc(n * sizeof(double));

    if (!A || !L || !U || !P || !b || !x) return 1;
    for (size_t i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            A->data[i][j] = 1.0 / (double)(i + j + 1);
            row_sum += A->data[i][j];
        }
        b[i] = row_sum;
    }
    printf("Résolution du système Hx = b (Hilbert n=%zu)\n", n);
    if (decompose_LU(A, L, U, P) == 0) {
        solve_LU(L, U, P, b, x, n);
        printf("\nSolution calculée x :\n");
        for (size_t i = 0; i < n; i++) printf("%f ", x[i]);
        double error = 0.0;
        for (size_t i = 0; i < n; i++) error += fabs(x[i] - 1.0);
        printf("\n\nErreur L1 absolue : %e\n", error);
    } else {
        printf("Erreur : Matrice singulière ou mal conditionnée.\n");
    }
    free_matrix(A); free_matrix(L); free_matrix(U);
    free(P); free(b); free(x);
    return 0;
}
