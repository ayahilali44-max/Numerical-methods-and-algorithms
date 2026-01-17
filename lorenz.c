#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef struct {
    int dim;
    double *data;
} Vector;
typedef void (*SystemFunc)(double, const Vector*, Vector*, void*);
typedef struct {
    double t_start;
    double t_end;
    double tol;
    double h_initial;
} SolverConfig;
typedef struct {
    int count;
    int capacity;
    double *t;
    Vector *y;
} Trajectory;
Vector create_vector(int dim) {
    Vector v = {dim, (double *)malloc(dim * sizeof(double))};
    return v;
}
void free_vector(Vector *v) {
    free(v->data);
}
void copy_vector(Vector dest, const Vector src) {
    for (int i = 0; i < src.dim; i++) dest.data[i] = src.data[i];
}
void vector_saxpy(Vector dest, const Vector a, const Vector b, double scale) {
    for (int i = 0; i < a.dim; i++) dest.data[i] = a.data[i] + b.data[i] * scale;
}
typedef struct {
    double sigma, rho, beta;
} LorenzParams;
void lorenz_system(double t, const Vector *y, Vector *deriv, void *params) {
    LorenzParams *p = (LorenzParams *)params;
    double x = y->data[0];
    double dy = y->data[1];
    double z = y->data[2];
    deriv->data[0] = p->sigma * (dy - x);
    deriv->data[1] = x * (p->rho - z) - dy;
    deriv->data[2] = x * dy - p->beta * z;
}
void save_step(Trajectory *tr, double t, Vector y) {
    if (tr->count >= tr->capacity) {
        tr->capacity *= 2;
        tr->t = realloc(tr->t, tr->capacity * sizeof(double));
        tr->y = realloc(tr->y, tr->capacity * sizeof(Vector));
    }
    tr->t[tr->count] = t;
    tr->y[tr->count] = create_vector(y.dim);
    copy_vector(tr->y[tr->count], y);
    tr->count++;
}
void rkf45_solve(SystemFunc func, Vector y0, SolverConfig cfg, void *params, Trajectory *tr) {
    int d = y0.dim;
    double t = cfg.t_start;
    double h = cfg.h_initial;
    Vector y = create_vector(d);
    copy_vector(y, y0);
    Vector k1 = create_vector(d), k2 = create_vector(d), k3 = create_vector(d);
    Vector k4 = create_vector(d), k5 = create_vector(d), k6 = create_vector(d);
    Vector tmp = create_vector(d);
    save_step(tr, t, y);
    while (t < cfg.t_end) {
        if (t + h > cfg.t_end) h = cfg.t_end - t;
        func(t, &y, &k1, params);
        vector_saxpy(tmp, y, k1, h * (1.0/4.0));
        func(t + h * (1.0/4.0), &tmp, &k2, params);
        for(int i=0; i<d; i++) tmp.data[i] = y.data[i] + h*(3.0/32.0*k1.data[i] + 9.0/32.0*k2.data[i]);
        func(t + h * (3.0/8.0), &tmp, &k3, params);
        for(int i=0; i<d; i++) tmp.data[i] = y.data[i] + h*(1932.0/2197.0*k1.data[i] - 7200.0/2197.0*k2.data[i] + 7296.0/2197.0*k3.data[i]);
        func(t + h * (12.0/13.0), &tmp, &k4, params);
        for(int i=0; i<d; i++) tmp.data[i] = y.data[i] + h*(439.0/216.0*k1.data[i] - 8.0*k2.data[i] + 3680.0/513.0*k3.data[i] - 845.0/4104.0*k4.data[i]);
        func(t + h, &tmp, &k5, params);
        for(int i=0; i<d; i++) tmp.data[i] = y.data[i] + h*(-8.0/27.0*k1.data[i] + 2.0*k2.data[i] - 3544.0/2565.0*k3.data[i] + 1859.0/4104.0*k4.data[i] - 11.0/40.0*k5.data[i]);
        func(t + h * (1.0/2.0), &tmp, &k6, params);
        double error = 0;
        for (int i = 0; i < d; i++) {
            double e_i = fabs(h * (1.0/360.0*k1.data[i] - 128.0/4275.0*k3.data[i] - 2197.0/75240.0*k4.data[i] + 1.0/50.0*k5.data[i] + 2.0/55.0*k6.data[i]));
            if (e_i > error) error = e_i;
        }
        if (error <= cfg.tol || h <= 1e-12) {
            t += h;
            for (int i = 0; i < d; i++) {
                y.data[i] += h * (16.0/135.0*k1.data[i] + 6656.0/12825.0*k3.data[i] + 28561.0/56430.0*k4.data[i] - 9.0/50.0*k5.data[i] + 2.0/55.0*k6.data[i]);
            }
            save_step(tr, t, y);
        }
        double delta = 0.84 * pow(cfg.tol / (error + 1e-18), 0.25);
        if (delta > 4.0) delta = 4.0;
        if (delta < 0.1) delta = 0.1;
        h *= delta;
    }
    free_vector(&k1);
    free_vector(&k2);
    free_vector(&k3);
    free_vector(&k4);
    free_vector(&k5);
    free_vector(&k6);
    free_vector(&tmp);
    free_vector(&y);
}
int main() {
    LorenzParams lp = {10.0, 28.0, 8.0 / 3.0};
    Vector y0 = create_vector(3);
    y0.data[0] = 1.0; y0.data[1] = 1.0; y0.data[2] = 1.0;
    SolverConfig cfg = {0.0, 50.0, 1e-6, 0.01};
    Trajectory tr = {0, 100, malloc(100 * sizeof(double)), malloc(100 * sizeof(Vector))};
    printf("Calcul de la trajectoire de Lorenz...\n");
    rkf45_solve(lorenz_system, y0, cfg, &lp, &tr);
    printf("Terminé. Points calculés : %d\n", tr.count);
    FILE *f = fopen("lorenz_output.csv", "w");
    fprintf(f, "t,x,y,z\n");
    for (int i = 0; i < tr.count; i++) {
        fprintf(f, "%f,%f,%f,%f\n", tr.t[i], tr.y[i].data[0], tr.y[i].data[1], tr.y[i].data[2]);
    }
    fclose(f);
    for (int i = 0; i < tr.count; i++) free_vector(&tr.y[i]);
    free(tr.t); free(tr.y); free_vector(&y0);
    return 0;
}
