#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef struct {
    int n;
    int nnz;
    double *values;
    int *col_ind;
    int *row_ptr;
} SparseMatrixCSR;
void free_sparse_matrix(SparseMatrixCSR *A) {
    free(A->values);
    free(A->col_ind);
    free(A->row_ptr);
}
void load_poisson_1d(int n, SparseMatrixCSR *A) {
    A->n = n;
    A->nnz = 3 * n - 2;
    A->values = (double *)malloc(A->nnz * sizeof(double));
    A->col_ind = (int *)malloc(A->nnz * sizeof(int));
    A->row_ptr = (int *)malloc((n + 1) * sizeof(int));
    if (!A->values || !A->col_ind || !A->row_ptr) exit(EXIT_FAILURE);
    int count = 0;
    for (int i = 0; i < n; i++) {
        A->row_ptr[i] = count;
        if (i > 0) {
            A->col_ind[count] = i - 1;
            A->values[count++] = -1.0;
        }
        A->col_ind[count] = i;
        A->values[count++] = 2.0;
        if (i < n - 1) {
            A->col_ind[count] = i + 1;
            A->values[count++] = -1.0;
        }
    }
    A->row_ptr[n] = count;
}
double dot_product(int n, const double *x, const double *y) {
    double res = 0.0;
    for (int i = 0; i < n; i++) res += x[i] * y[i];
    return res;
}
void spmv_csr(const SparseMatrixCSR *A, const double *x, double *y) {
    for (int i = 0; i < A->n; i++) {
        double sum = 0.0;
        for (int k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
            sum += A->values[k] * x[A->col_ind[k]];
        }
        y[i] = sum;
    }
}
int conjugate_gradient(const SparseMatrixCSR *A, const double *b, double *x, double tol, int max_iter) {
    int n = A->n;
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *Ap = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) x[i] = 0.0;
    for (int i = 0; i < n; i++) r[i] = b[i];
    for (int i = 0; i < n; i++) p[i] = r[i];
    double r_old_dot = dot_product(n, r, r);
    double b_norm = sqrt(dot_product(n, b, b));
    int k = 0;
    while (k < max_iter) {
        spmv_csr(A, p, Ap);
        double pAp = dot_product(n, p, Ap);
        double alpha = r_old_dot / pAp;
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        double r_new_dot = dot_product(n, r, r);
        double residual = sqrt(r_new_dot) / b_norm;
        if (residual < tol) {
            k++;
            break;
        }
        double beta = r_new_dot / r_old_dot;
        for (int i = 0; i < n; i++) {
            p[i] = r[i] + beta * p[i];
        }
        r_old_dot = r_new_dot;
        k++;
        if (k % 100 == 0) printf("Iteration %d, Residu relatif: %e\n", k, residual);
    }
    free(r); free(p); free(Ap);
    return k;
}
int main() {
    int n = 1000;
    SparseMatrixCSR A;
    load_poisson_1d(n, &A);
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) b[i] = 1.0;
    printf("Résolution du système Poisson 1D (N=%d) par Gradient Conjugué...\n", n);
    int iters = conjugate_gradient(&A, b, x, 1e-10, 2000);
    printf("\nConvergence atteinte en %d itérations.\n", iters);
    printf("Valeurs de x (échantillon) : x[0]=%.4f, x[n/2]=%.4f, x[n-1]=%.4f\n", x[0], x[n/2], x[n-1]);
    free_sparse_matrix(&A);
    free(b);
    free(x);
    return 0;
}
