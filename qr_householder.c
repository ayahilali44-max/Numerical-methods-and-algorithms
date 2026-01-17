#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
typedef struct {
    int rows;
    int cols;
    double *data;
} Matrix;
Matrix allocate_matrix(int rows, int cols) {
    Matrix M;
    M.rows = rows;
    M.cols = cols;
    M.data = (double *)malloc(rows * cols * sizeof(double));
    if (M.data == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire.\n");
        exit(EXIT_FAILURE);
    }
    return M;
}
void free_matrix(Matrix *M) {
    if (M->data != NULL) {
        free(M->data);
        M->data = NULL;
    }
}
double get(Matrix M, int i, int j) { return M.data[i * M.cols + j]; }
void set(Matrix M, int i, int j, double val) {
    M.data[i * M.cols + j] = val;
}
void print_matrix(const char *name, Matrix M) {
    printf("Matrice %s (%d x %d):\n", name, M.rows, M.cols);
    for (int i = 0; i < M.rows; i++) {
        for (int j = 0; j < M.cols; j++) {
            printf("%10.6f ", get(M, i, j));
        }
        printf("\n");
    }
    printf("\n");
}
double frobenius_norm_diff(Matrix A, Matrix B) {
    double sum = 0;
    for (int i = 0; i < A.rows * A.cols; i++) {
        double diff = A.data[i] - B.data[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}
void apply_householder_to_matrix(Matrix M, double *v, int start_row, int start_col) {
    int m = M.rows;
    int n = M.cols;
    for (int j = start_col; j < n; j++) {
        double dot = 0;
        for (int i = start_row; i < m; i++) {
            dot += v[i - start_row] * get(M, i, j);
        }
        for (int i = start_row; i < m; i++) {
            double val = get(M, i, j) - 2.0 * v[i - start_row] * dot;
            set(M, i, j, val);
        }
    }
}
void decompose_qr(Matrix A, Matrix *Q, Matrix *R) {
    int m = A.rows;
    int n = A.cols;
    memcpy(R->data, A.data, m * n * sizeof(double));
    memset(Q->data, 0, m * m * sizeof(double));
    for (int i = 0; i < m; i++) set(*Q, i, i, 1.0);
    for (int k = 0; k < n && k < m - 1; k++) {
        int dim = m - k;
        double *x = (double *)malloc(dim * sizeof(double));
        double *v = (double *)malloc(dim * sizeof(double));
        double norm_x = 0;
        for (int i = 0; i < dim; i++) {
            x[i] = get(*R, i + k, k);
            norm_x += x[i] * x[i];
        }
        norm_x = sqrt(norm_x);
        double s = (x[0] >= 0) ? 1.0 : -1.0;
        double v_norm_sq = 0;

        v[0] = x[0] + s * norm_x;
        v_norm_sq += v[0] * v[0];
        for (int i = 1; i < dim; i++) {
            v[i] = x[i];
            v_norm_sq += v[i] * v[i];
        }
        double v_norm = sqrt(v_norm_sq);
        if (v_norm > 1e-15) {
            for (int i = 0; i < dim; i++) v[i] /= v_norm;
            apply_householder_to_matrix(*R, v, k, k);
            for (int i = 0; i < m; i++) {
                double dot = 0;
                for (int j = k; j < m; j++) {
                    dot += get(*Q, i, j) * v[j - k];
                }
                for (int j = k; j < m; j++) {
                    double val = get(*Q, i, j) - 2.0 * dot * v[j - k];
                    set(*Q, i, j, val);
                }
            }
        }
        free(x);
        free(v);
    }
}
int main() {
    int n = 6;
    Matrix A = allocate_matrix(n, n);
    Matrix Q = allocate_matrix(n, n);
    Matrix R = allocate_matrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            set(A, i, j, 1.0 / (i + j + 1));
        }
    }
    print_matrix("A (Hilbert 6x6)", A);
    decompose_qr(A, &Q, &R);
    print_matrix("Q (Orthogonale)", Q);
    print_matrix("R (Triangulaire supérieure)", R);
    Matrix QR = allocate_matrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) sum += get(Q, i, k) * get(R, k, j);
            set(QR, i, j, sum);
        }
    }
    printf("Erreur de reconstruction ||A - QR||_F : %e\n", frobenius_norm_diff(A, QR));
    Matrix QtQ = allocate_matrix(n, n);
    Matrix I = allocate_matrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) sum += get(Q, k, i) * get(Q, k, j);
            set(QtQ, i, j, sum);
            set(I, i, j, (i == j) ? 1.0 : 0.0);
        }
    }
    printf("Erreur d'orthogonalité ||Q^T*Q - I||_F : %e\n", frobenius_norm_diff(QtQ, I));
    free_matrix(&A);
    free_matrix(&Q);
    free_matrix(&R);
    free_matrix(&QR);
    free_matrix(&QtQ);
    free_matrix(&I);
    return 0;
}
