// CPU-only interpolating cubic B-spline image pull and analytic world-mm gradient.
// The Python caller supplies SciPy-prefiltered, nearest-padded coefficients.
// Source [D,X,Y,Z], u [D,3,X,Y,Z]; no trainable source intensities.
#include <cmath>
#include <cstdint>
#include <cstddef>

static inline void weights(double t, double *w, double *g) {
    const double s = 1.0-t;
    w[0]=s*s*s/6.; w[1]=(3*t*t*t-6*t*t+4)/6.;
    w[2]=(-3*t*t*t+3*t*t+3*t+1)/6.; w[3]=t*t*t/6.;
    g[0]=-.5*s*s; g[1]=1.5*t*t-2*t;
    g[2]=-1.5*t*t+t+.5; g[3]=.5*t*t;
}

extern "C" void cubic_pull(const double* coeff, const double* src,
        const double* u, const double* inv, int directions, int nx, int ny, int nz,
        int pad, int threads, bool need_gradient,
        double* out, double* grad, uint8_t* valid) {
    const int px=nx+2*pad, py=ny+2*pad, pz=nz+2*pad;
    const int64_t n=int64_t(nx)*ny*nz, pn=int64_t(px)*py*pz;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t index=0; index<int64_t(directions)*n; ++index) {
        const int d=int(index/n); const int64_t p=index%n;
        const int x=int(p/(ny*nz)), y=int((p/nz)%ny), z=int(p%nz);
        double v[3]={u[(d*3+0)*n+p],u[(d*3+1)*n+p],u[(d*3+2)*n+p]};
        double q[3]={double(x),double(y),double(z)};
        for (int i=0;i<3;++i) for(int j=0;j<3;++j) q[i]+=inv[i*3+j]*v[j];
        const bool ok=q[0]>=0 && q[0]<=nx-1 && q[1]>=0 && q[1]<=ny-1 && q[2]>=0 && q[2]<=nz-1;
        valid[index]=ok;
        if (!ok) {
            // Match the original fallback, NEVER discard a voxel.
            out[index]=src[index];
            if(need_gradient) for(int j=0;j<3;++j) grad[(d*3+j)*n+p]=0.;
            continue;
        }
        double w[3][4],g[3][4];int start[3];
        for(int j=0;j<3;++j) {
            double f=std::floor(q[j]); start[j]=int(f)+pad-1;
            weights(q[j]-f,w[j],g[j]);
        }
        double value=0.,gx=0.,gy=0.,gz=0.;
        for(int i=0;i<4;++i) for(int j=0;j<4;++j) for(int k=0;k<4;++k) {
            const double c=coeff[int64_t(d)*pn+(int64_t(start[0]+i)*py+start[1]+j)*pz+start[2]+k];
            value+=c*w[0][i]*w[1][j]*w[2][k];
            if(need_gradient) {
                gx+=c*g[0][i]*w[1][j]*w[2][k];
                gy+=c*w[0][i]*g[1][j]*w[2][k];
                gz+=c*w[0][i]*w[1][j]*g[2][k];
            }
        }
        out[index]=value;
        if(need_gradient) for(int j=0;j<3;++j)
            grad[(d*3+j)*n+p]=gx*inv[j]+gy*inv[3+j]+gz*inv[6+j];
    }
}
