/*
 * dbn_tb.cpp  —  DBN testbench: software inference, IP core, AUC comparison
 *
 * Three paths compared:
 *   Py  — Python T3 reference RMSE scores (loaded from rmse_t3.bin)
 *   SW  — C software inference (dbn_forward, stage-validated)
 *   IP  — HLS kernel (dbn_inout) called directly
 *
 * Loads fixed_export/<scene>/ produced by inspect_ptq.py.
 *
 * Compile for host testing (sw only, no IP):
 *   g++ -O2 -I$XILINX_HLS/include -o dbn_tb dbn_tb.cpp -lm
 *
 * Vitis HLS csim:
 *   add_files dbn_inout.cpp  -cflags "-std=c++14"
 *   add_files -tb dbn_tb.cpp -cflags "-std=c++14"
 *
 * Usage:
 *   ./dbn_tb fixed_export/abu-airport-1
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "ap_int.h"

/* Include the IP kernel header when building under Vitis HLS */
#include "dbn_inout.hpp"

/* ── SW-path ap_int aliases ──────────────────────────────────────────────── */
typedef ap_int<8>   w8_t;
typedef ap_int<32>  b32_t;
typedef ap_int<64>  acc64_t;
typedef ap_uint<32> m32_t;
typedef ap_int<64>  wide_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * Config + Model structs
 * ═════════════════════════════════════════════════════════════════════════ */
typedef struct {
    char scene[64];
    int  height, width, n_bands, n_pixels, n_input, n_hidden;
    float    input_scale_f32;  int input_zp;
    uint32_t enc_mult;         int enc_shift;   int enc_act_zp;
    int      sigmoid_zp;
    uint32_t dec_mult;         int dec_shift;
    float    output_scale_f32; int output_zp;
} dbn_config_t;

typedef struct {
    dbn_config_t cfg;
    int8_t  *enc_w;   int32_t *enc_b;
    int8_t  *dec_w;   int32_t *dec_b;
    int8_t   sigmoid_lut[256];
    float   *scaler_mean;   float *scaler_scale;
} dbn_model_t;

/* ═══════════════════════════════════════════════════════════════════════════
 * JSON / file helpers
 * ═════════════════════════════════════════════════════════════════════════ */
static const char *jval(const char *buf, const char *key) {
    char needle[128]; snprintf(needle,sizeof(needle),"\"%s\"",key);
    const char *p=strstr(buf,needle); if(!p) return NULL;
    p+=strlen(needle);
    while(*p==' '||*p=='\t'||*p==':') p++;
    return p;
}
static int jint   (const char*b,const char*k,int     *o){const char*p=jval(b,k);if(!p){fprintf(stderr,"[cfg] missing: %s\n",k);return -1;}*o=(int)strtol(p,NULL,10);return 0;}
static int juint32(const char*b,const char*k,uint32_t*o){const char*p=jval(b,k);if(!p){fprintf(stderr,"[cfg] missing: %s\n",k);return -1;}*o=(uint32_t)strtoul(p,NULL,10);return 0;}
static int jfloat (const char*b,const char*k,float   *o){const char*p=jval(b,k);if(!p){fprintf(stderr,"[cfg] missing: %s\n",k);return -1;}*o=(float)strtod(p,NULL);return 0;}
static int jstr   (const char*b,const char*k,char*o,int n){const char*p=jval(b,k);if(!p||*p!='"')return -1;p++;int i=0;while(*p&&*p!='"'&&i<n-1)o[i++]=*p++;o[i]='\0';return 0;}

static char *read_text_file(const char *path) {
    FILE *f=fopen(path,"r"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long sz=ftell(f); rewind(f);
    char *buf=(char*)malloc(sz+1); fread(buf,1,sz,f); buf[sz]='\0'; fclose(f);
    return buf;
}

static long read_binary(const char *path, void **buf, size_t esz) {
    FILE *f=fopen(path,"rb"); if(!f){fprintf(stderr,"Cannot open %s\n",path);return -1;}
    fseek(f,0,SEEK_END); long nb=ftell(f); rewind(f);
    *buf=malloc(nb>0?nb:1);
    size_t nr=fread(*buf,1,nb,f); fclose(f);
    if((long)nr!=nb){fprintf(stderr,"Short read %s\n",path);free(*buf);*buf=NULL;return -1;}
    return nb/(long)esz;
}

static int load_config(const char *dir, dbn_config_t *c) {
    char path[512]; snprintf(path,sizeof(path),"%s/config.json",dir);
    char *buf=read_text_file(path);
    if(!buf){fprintf(stderr,"Cannot open %s\n",path);return -1;}
    int e=0;
    jstr(buf,"scene",c->scene,sizeof(c->scene));
    e|=jint(buf,"height",&c->height);      e|=jint(buf,"width",&c->width);
    e|=jint(buf,"n_bands",&c->n_bands);    e|=jint(buf,"n_pixels",&c->n_pixels);
    e|=jint(buf,"n_input",&c->n_input);    e|=jint(buf,"n_hidden",&c->n_hidden);
    e|=jfloat(buf,"input_scale_f32",&c->input_scale_f32);
    e|=jint(buf,"input_zp",&c->input_zp);
    e|=juint32(buf,"enc_mult",&c->enc_mult);
    e|=jint(buf,"enc_shift",&c->enc_shift);
    e|=jint(buf,"enc_act_zp",&c->enc_act_zp);
    e|=jint(buf,"sigmoid_zp",&c->sigmoid_zp);
    e|=juint32(buf,"dec_mult",&c->dec_mult);
    e|=jint(buf,"dec_shift",&c->dec_shift);
    e|=jfloat(buf,"output_scale_f32",&c->output_scale_f32);
    e|=jint(buf,"output_zp",&c->output_zp);
    free(buf); return e;
}

static int load_model(const char *dir, dbn_model_t *m) {
    char path[512];
    if(load_config(dir,&m->cfg)!=0) return -1;
    const dbn_config_t *c=&m->cfg;
    printf("  Scene   : %s  (%dx%dx%d  n_input=%d  n_hidden=%d)\n",
           c->scene,c->height,c->width,c->n_bands,c->n_input,c->n_hidden);
#define LOAD(fn,field,ctype) \
    snprintf(path,sizeof(path),"%s/" fn,dir); \
    if(read_binary(path,(void**)&m->field,sizeof(ctype))<0) return -1;
    LOAD("enc_w.bin",enc_w,int8_t)   LOAD("enc_b.bin",enc_b,int32_t)
    LOAD("dec_w.bin",dec_w,int8_t)   LOAD("dec_b.bin",dec_b,int32_t)
    LOAD("scaler_mean.bin",scaler_mean,float)
    LOAD("scaler_scale.bin",scaler_scale,float)
#undef LOAD
    snprintf(path,sizeof(path),"%s/sigmoid_lut.bin",dir);
    FILE *lf=fopen(path,"rb"); if(!lf) return -1;
    fread(m->sigmoid_lut,1,256,lf); fclose(lf);
    return 0;
}

static void free_model(dbn_model_t *m) {
    free(m->enc_w); free(m->enc_b); free(m->dec_w); free(m->dec_b);
    free(m->scaler_mean); free(m->scaler_scale);
    memset(m,0,sizeof(*m));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Scaler
 * ═════════════════════════════════════════════════════════════════════════ */
static void scaler_apply(const float *raw, int n,
                         const float *mean, const float *scale, float *out) {
    for(int i=0;i<n;i++) out[i]=(raw[i]-mean[i])/scale[i];
}
static void scaler_inverse(const float *norm, int n,
                           const float *mean, const float *scale, float *out) {
    for(int i=0;i<n;i++) out[i]=norm[i]*scale[i]+mean[i];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SW inference layers
 * ═════════════════════════════════════════════════════════════════════════ */
static void quantise_input(const float *px, int ni, float scale, int zp, w8_t *out) {
    for(int i=0;i<ni;i++){
        int q=(int)roundf(px[i]/scale)+zp;
        if(q>127)q=127; else if(q<-128)q=-128;
        out[i]=(w8_t)q;
    }
}
static void encoder_matmul(const w8_t *inp_q, int ni,
                            const w8_t *enc_w, const b32_t *enc_b, int nh,
                            int zp, acc64_t *acc) {
    for(int h=0;h<nh;h++){
        acc64_t a=(acc64_t)enc_b[h];
        for(int i=0;i<ni;i++) a+=(acc64_t)enc_w[h*ni+i]*((acc64_t)inp_q[i]-(acc64_t)zp);
        acc[h]=a;
    }
}
static w8_t requantise(acc64_t acc, m32_t mult, int shift, int zp) {
    wide_t p=(wide_t)acc*(wide_t)mult;
    wide_t r=(p+((wide_t)1<<(shift-1)))>>shift;
    wide_t q=r+(wide_t)zp;
    if(q>127)q=127; else if(q<-128)q=-128;
    return (w8_t)q;
}
static void encoder_requant(const acc64_t *acc, int nh,
                             m32_t mult, int shift, int zp, w8_t *out) {
    for(int h=0;h<nh;h++) out[h]=requantise(acc[h],mult,shift,zp);
}
static void sigmoid_lut_apply(const w8_t *enc_req, int nh,
                               const w8_t *lut, w8_t *hidden) {
    for(int h=0;h<nh;h++) hidden[h]=lut[(int)enc_req[h]+128];
}
static void decoder_matmul(const w8_t *hidden, int nh,
                            const w8_t *dec_w, const b32_t *dec_b, int ni,
                            int zp, acc64_t *acc) {
    for(int i=0;i<ni;i++){
        acc64_t a=(acc64_t)dec_b[i];
        for(int h=0;h<nh;h++) a+=(acc64_t)dec_w[i*nh+h]*((acc64_t)hidden[h]-(acc64_t)zp);
        acc[i]=a;
    }
}
static void decoder_requant(const acc64_t *acc, int ni,
                             m32_t mult, int shift, int zp, w8_t *out) {
    for(int i=0;i<ni;i++) out[i]=requantise(acc[i],mult,shift,zp);
}
static void dequantise(const w8_t *out_q, int ni, float scale, int zp, float *out) {
    for(int i=0;i<ni;i++) out[i]=((float)(int)out_q[i]-(float)zp)*scale;
}

static void dbn_forward(const float *px_norm, const dbn_model_t *m, float *out_f32,
                        w8_t *dbg_inp_q, acc64_t *dbg_enc_acc,
                        w8_t *dbg_enc_req, w8_t *dbg_hidden,
                        acc64_t *dbg_dec_acc, w8_t *dbg_out_q)
{
    const dbn_config_t *c=&m->cfg;
    int ni=c->n_input, nh=c->n_hidden;
    w8_t    inp_q[256]; acc64_t enc_acc[32]; w8_t enc_req[32];
    w8_t    hidden[32]; acc64_t dec_acc[256]; w8_t out_q[256];
    const w8_t  *ew=(const w8_t*)m->enc_w; const b32_t *eb=(const b32_t*)m->enc_b;
    const w8_t  *dw=(const w8_t*)m->dec_w; const b32_t *db=(const b32_t*)m->dec_b;
    quantise_input   (px_norm,ni,c->input_scale_f32,c->input_zp,inp_q);
    encoder_matmul   (inp_q,ni,ew,eb,nh,c->input_zp,enc_acc);
    encoder_requant  (enc_acc,nh,(m32_t)c->enc_mult,c->enc_shift,c->enc_act_zp,enc_req);
    sigmoid_lut_apply(enc_req,nh,(const w8_t*)m->sigmoid_lut,hidden);
    decoder_matmul   (hidden,nh,dw,db,ni,c->sigmoid_zp,dec_acc);
    decoder_requant  (dec_acc,ni,(m32_t)c->dec_mult,c->dec_shift,c->output_zp,out_q);
    dequantise       (out_q,ni,c->output_scale_f32,c->output_zp,out_f32);
    if(dbg_inp_q)   memcpy(dbg_inp_q,  inp_q,  ni*sizeof(w8_t));
    if(dbg_enc_acc) memcpy(dbg_enc_acc,enc_acc, nh*sizeof(acc64_t));
    if(dbg_enc_req) memcpy(dbg_enc_req,enc_req, nh*sizeof(w8_t));
    if(dbg_hidden)  memcpy(dbg_hidden, hidden,  nh*sizeof(w8_t));
    if(dbg_dec_acc) memcpy(dbg_dec_acc,dec_acc, ni*sizeof(acc64_t));
    if(dbg_out_q)   memcpy(dbg_out_q,  out_q,   ni*sizeof(w8_t));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Stage-by-stage pixel validation (SW vs Python T3)
 * ═════════════════════════════════════════════════════════════════════════ */
typedef struct {
    int8_t  *inp_q; int64_t *enc_acc; int8_t *enc_req;
    int8_t  *hidden; int64_t *dec_acc; int8_t *out_q; float *out_f32;
    float    rmse_norm, rmse_raw;
    int      px_idx;
    char     label[32];
} ref_pixel_t;

static int load_ref_pixel(const char *px_dir, ref_pixel_t *r, int ni, int nh) {
    char path[512]; char *buf;
    snprintf(path,sizeof(path),"%s/info.json",px_dir);
    buf=read_text_file(path); if(!buf) return -1;
    jstr(buf,"label",r->label,sizeof(r->label));
    jint(buf,"px_idx",&r->px_idx);
    jfloat(buf,"rmse_norm",&r->rmse_norm);
    jfloat(buf,"rmse_raw", &r->rmse_raw);
    free(buf);
#define LOADR(fn,field,ctype) \
    snprintf(path,sizeof(path),"%s/" fn,px_dir); \
    if(read_binary(path,(void**)&r->field,sizeof(ctype))<0) return -1;
    LOADR("inp_q.bin",  inp_q,  int8_t)  LOADR("enc_acc.bin",enc_acc,int64_t)
    LOADR("enc_req.bin",enc_req,int8_t)  LOADR("hidden.bin", hidden, int8_t)
    LOADR("dec_acc.bin",dec_acc,int64_t) LOADR("out_q.bin",  out_q,  int8_t)
    LOADR("out_f32.bin",out_f32,float)
#undef LOADR
    return 0;
}
static void free_ref_pixel(ref_pixel_t *r) {
    free(r->inp_q); free(r->enc_acc); free(r->enc_req); free(r->hidden);
    free(r->dec_acc); free(r->out_q); free(r->out_f32);
    memset(r,0,sizeof(*r));
}

static int validate_pixel(const dbn_model_t *m, const ref_pixel_t *ref,
                           const float *image_raw)
{
    const dbn_config_t *c=&m->cfg;
    int ni=c->n_input, nh=c->n_hidden;
    const float *px_raw=image_raw+(long)ref->px_idx*ni;
    float px_norm[256];
    scaler_apply(px_raw,ni,m->scaler_mean,m->scaler_scale,px_norm);

    w8_t    dbg_inp_q[256]; acc64_t dbg_enc_acc[32]; w8_t dbg_enc_req[32];
    w8_t    dbg_hidden[32]; acc64_t dbg_dec_acc[256]; w8_t dbg_out_q[256];
    float   out_f32[256];
    dbn_forward(px_norm,m,out_f32,
                dbg_inp_q,dbg_enc_acc,dbg_enc_req,
                dbg_hidden,dbg_dec_acc,dbg_out_q);

    float rmse=0.0f;
    for(int i=0;i<ni;i++){float d=px_norm[i]-out_f32[i];rmse+=d*d;}
    rmse=sqrtf(rmse/ni);

    printf("\n  ── Pixel: %s (idx=%d) ──────────────────────────\n",ref->label,ref->px_idx);
    printf("  RMSE_norm  SW=%.6f  Py=%.6f  diff=%.2e  %s\n",
           rmse,ref->rmse_norm,fabsf(rmse-ref->rmse_norm),
           fabsf(rmse-ref->rmse_norm)<1e-4f?"OK":"DIFF");

    int mm=0;
#define CMP_INT8(stage,cb,rb,n) { \
    int _mm=0,_mx=0; \
    for(int _i=0;_i<(n);_i++){int _d=abs((int)(int8_t)(cb)[_i]-(int)(rb)[_i]);if(_d>0){_mm++;if(_d>_mx)_mx=_d;}} \
    printf("  %-10s  mm=%d/%d  maxDiff=%d  %s\n",stage,_mm,(n),_mx,_mm==0?"OK":"MISMATCH"); mm+=_mm;}
#define CMP_INT64(stage,cb,rb,n) { \
    int _mm=0; long _mx=0; \
    for(int _i=0;_i<(n);_i++){long _d=llabs((long)(cb)[_i]-(long)(rb)[_i]);if(_d>0){_mm++;if(_d>_mx)_mx=_d;}} \
    printf("  %-10s  mm=%d/%d  maxDiff=%ld  %s\n",stage,_mm,(n),_mx,_mm==0?"OK":"MISMATCH"); mm+=_mm;}
#define CMP_F32(stage,cb,rb,n) { \
    int _mm=0; float _mx=0; \
    for(int _i=0;_i<(n);_i++){float _d=fabsf((cb)[_i]-(rb)[_i]);if(_d>1e-6f){_mm++;if(_d>_mx)_mx=_d;}} \
    printf("  %-10s  mm=%d/%d  maxDiff=%.2e  %s\n",stage,_mm,(n),_mx,_mm==0?"OK":"MISMATCH"); mm+=_mm;}

    CMP_INT8 ("inp_q",  dbg_inp_q,  ref->inp_q,  ni)
    CMP_INT64("enc_acc",dbg_enc_acc,ref->enc_acc, nh)
    CMP_INT8 ("enc_req",dbg_enc_req,ref->enc_req, nh)
    CMP_INT8 ("hidden", dbg_hidden, ref->hidden,  nh)
    CMP_INT64("dec_acc",dbg_dec_acc,ref->dec_acc, ni)
    CMP_INT8 ("out_q",  dbg_out_q,  ref->out_q,   ni)
    CMP_F32  ("out_f32",out_f32,    ref->out_f32, ni)
#undef CMP_INT8
#undef CMP_INT64
#undef CMP_F32
    printf("  %s\n",mm==0?"ALL STAGES MATCH":"STAGE MISMATCHES");
    return mm;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * AUC  (trapezoid rule)
 * ═════════════════════════════════════════════════════════════════════════ */
typedef struct { float score; int label; } scored_t;
static int cmp_scored_desc(const void *a, const void *b) {
    const scored_t *sa=(const scored_t*)a,*sb=(const scored_t*)b;
    if(sb->score>sa->score)return 1; if(sb->score<sa->score)return -1; return 0;
}
static float compute_auc(const float *scores, const uint8_t *labels, int n) {
    long np=0,ng=0;
    for(int i=0;i<n;i++){if(labels[i])np++;else ng++;}
    if(!np||!ng)return -1.0f;
    scored_t *s=(scored_t*)malloc(n*sizeof(scored_t));
    for(int i=0;i<n;i++){s[i].score=scores[i];s[i].label=labels[i];}
    qsort(s,n,sizeof(scored_t),cmp_scored_desc);
    double auc=0.0; long tp=0,fp=0,ptp=0,pfp=0;
    for(int i=0;i<n;i++){
        if(s[i].label)tp++;else fp++;
        if(i==n-1||s[i].score!=s[i+1].score){
            auc+=0.5*((double)(tp+ptp)/np)*((double)(fp-pfp)/ng);
            ptp=tp; pfp=fp;
        }
    }
    free(s); return (float)auc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SW full-image pass
 * ═════════════════════════════════════════════════════════════════════════ */
static float *run_sw_image(const dbn_model_t *m, const float *image_raw)
{
    const dbn_config_t *c=&m->cfg;
    int ni=c->n_input, N=c->n_pixels;
    float *rmse=(float*)malloc(N*sizeof(float));
    float px_norm[256], out_norm[256], out_raw[256];
    for(int px=0;px<N;px++){
        const float *raw=image_raw+(long)px*ni;
        scaler_apply(raw,ni,m->scaler_mean,m->scaler_scale,px_norm);
        dbn_forward(px_norm,m,out_norm,NULL,NULL,NULL,NULL,NULL,NULL);
        scaler_inverse(out_norm,ni,m->scaler_mean,m->scaler_scale,out_raw);
        float mse=0.0f;
        for(int i=0;i<ni;i++){float d=raw[i]-out_raw[i];mse+=d*d;}
        rmse[px]=sqrtf(mse/ni);
    }
    return rmse;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * IP core full-image pass
 *
 * Prepares flat arrays in the layout dbn_inout() expects, calls the kernel,
 * returns the RMSE score array.
 * ═════════════════════════════════════════════════════════════════════════ */
static float *run_ip_image(const dbn_model_t *m, const float *image_raw)
{
    const dbn_config_t *c=&m->cfg;
    int ni=c->n_input, nh=c->n_hidden, N=c->n_pixels;

    /* ── Prepare weight arrays in IP layout ─────────────────────────────
     * SW stores enc_w as [n_hidden, n_input] row-major.
     * The IP kernel uses the same layout — copy cast to dat_tw1 (ap_int<8>). */
    static dat_tw1 W_enc[MAX_CODES * MAX_BANDS];
    static dat_tw2 W_dec[MAX_BANDS * MAX_CODES];
    static dat_tb1 B_enc[MAX_CODES];
    static dat_tb2 B_dec[MAX_BANDS];
    static dat_tc1 LUT  [LUT_SIZE];
    static float   sc_mean [MAX_BANDS];
    static float   sc_scale[MAX_BANDS];
    static float   hsi_in  [BUFFER_SIZE];
    static float   hsi_out [BUFFER_OUT];

    for(int i=0;i<nh*ni;i++) W_enc[i]=(dat_tw1)m->enc_w[i];
    for(int i=0;i<ni*nh;i++) W_dec[i]=(dat_tw2)m->dec_w[i];
    for(int i=0;i<nh;i++)    B_enc[i]=(dat_tb1)m->enc_b[i];
    for(int i=0;i<ni;i++)    B_dec[i]=(dat_tb2)m->dec_b[i];
    for(int i=0;i<LUT_SIZE;i++) LUT[i]=(dat_tc1)m->sigmoid_lut[i];
    for(int i=0;i<ni;i++){sc_mean[i]=m->scaler_mean[i]; sc_scale[i]=m->scaler_scale[i];}

    /* Flat image: [N * ni] floats */
    memcpy(hsi_in, image_raw, (long)N*ni*sizeof(float));

    /* ── Call IP kernel ─────────────────────────────────────────────── */
    dbn_inout(
        W_enc, B_enc, W_dec, B_dec, LUT, sc_mean, sc_scale,
        hsi_in, hsi_out,
        c->height, c->width,
        c->input_scale_f32, c->input_zp,
        c->enc_mult, c->enc_shift, c->enc_act_zp,
        c->sigmoid_zp,
        c->dec_mult, c->dec_shift,
        c->output_scale_f32, c->output_zp
    );

    float *rmse=(float*)malloc(N*sizeof(float));
    memcpy(rmse, hsi_out, N*sizeof(float));
    return rmse;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Comparison helper
 * ═════════════════════════════════════════════════════════════════════════ */
static void compare_rmse(const char *label_a, const float *a,
                         const char *label_b, const float *b,
                         int N, float tol)
{
    float max_diff=0.0f, mean_diff=0.0f; int mm=0;
    float mna=a[0],mxa=a[0],sma=0.0f;
    float mnb=b[0],mxb=b[0],smb=0.0f;
    for(int i=0;i<N;i++){
        float d=fabsf(a[i]-b[i]);
        if(d>tol)mm++;
        if(d>max_diff)max_diff=d;
        mean_diff+=d;
        if(a[i]<mna)mna=a[i]; if(a[i]>mxa)mxa=a[i]; sma+=a[i];
        if(b[i]<mnb)mnb=b[i]; if(b[i]>mxb)mxb=b[i]; smb+=b[i];
    }
    mean_diff/=N;
    printf("  %-4s  min=%.4f  max=%.4f  mean=%.4f\n",label_a,mna,mxa,sma/N);
    printf("  %-4s  min=%.4f  max=%.4f  mean=%.4f\n",label_b,mnb,mxb,smb/N);
    printf("  %s vs %s:  maxDiff=%.4e  meanDiff=%.4e  mm(>%.3f)=%d/%d  %s\n",
           label_a,label_b,max_diff,mean_diff,tol,mm,N,mm==0?"OK":"DIFF");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main
 * ═════════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    if(argc<2){
        fprintf(stderr,"Usage: %s <scene_dir>\n",argv[0]); return 1;
    }
    const char *dir=argv[1];

    /* ── Load model ─────────────────────────────────────────────────────── */
    printf("\n══════════════════════════════════════════════════\n");
    printf("  Loading model\n");
    printf("══════════════════════════════════════════════════\n");
    dbn_model_t model; memset(&model,0,sizeof(model));
    if(load_model(dir,&model)!=0){fprintf(stderr,"Load FAILED\n");return 1;}
    const dbn_config_t *c=&model.cfg;
    int ni=c->n_input, nh=c->n_hidden, N=c->n_pixels;

    /* ── Load validation data ────────────────────────────────────────────── */
    printf("\n══════════════════════════════════════════════════\n");
    printf("  Loading validation data\n");
    printf("══════════════════════════════════════════════════\n");
    char path[512];
    float   *image_raw=NULL, *rmse_py=NULL;
    uint8_t *gt_mask=NULL;
#define LOAD_VAL(fn,ptr,ctype) \
    snprintf(path,sizeof(path),"%s/validation/" fn,dir); \
    {long _n=read_binary(path,(void**)&ptr,sizeof(ctype)); \
     if(_n<0){fprintf(stderr,"Failed: %s\n",fn);return 1;} \
     printf("  %-22s %ld elements\n",fn,_n);}
    LOAD_VAL("image_raw.bin",image_raw,float)
    LOAD_VAL("gt_mask.bin",  gt_mask,  uint8_t)
    LOAD_VAL("rmse_t3.bin",  rmse_py,  float)
#undef LOAD_VAL

    /* ── Stage validation (SW vs Py) ─────────────────────────────────────── */
    printf("\n══════════════════════════════════════════════════\n");
    printf("  Stage-by-stage validation  SW vs Python T3\n");
    printf("══════════════════════════════════════════════════\n");
    const char *px_dirs[]={"px_centre","px_anomaly"};
    int stage_mm=0;
    for(int p=0;p<2;p++){
        snprintf(path,sizeof(path),"%s/validation/%s",dir,px_dirs[p]);
        char ipath[512]; snprintf(ipath,sizeof(ipath),"%s/info.json",path);
        FILE *chk=fopen(ipath,"r");
        if(!chk){printf("  Skipping %s\n",px_dirs[p]);continue;}
        fclose(chk);
        ref_pixel_t ref; memset(&ref,0,sizeof(ref));
        if(load_ref_pixel(path,&ref,ni,nh)==0){
            stage_mm+=validate_pixel(&model,&ref,image_raw);
            free_ref_pixel(&ref);
        }
    }

    /* ── SW full image ───────────────────────────────────────────────────── */
    printf("\n══════════════════════════════════════════════════\n");
    printf("  SW full-image RMSE  (%d pixels)\n",N);
    printf("══════════════════════════════════════════════════\n");
    float *rmse_sw=run_sw_image(&model,image_raw);
    compare_rmse("SW", rmse_sw,  "Py", rmse_py,  N, 0.01f);
    /* Silenced — reuse variable names below */
    (void)rmse_sw;

    /* Rerun for clean variable scope */
    float *rmse_c=rmse_sw;  /* alias */

    /* ── IP core full image ──────────────────────────────────────────────── */
    printf("\n══════════════════════════════════════════════════\n");
    printf("  IP core full-image RMSE  (%d pixels)\n",N);
    printf("══════════════════════════════════════════════════\n");
    float *rmse_ip=run_ip_image(&model,image_raw);

    /* IP vs Py */
    printf("\n  IP vs Py:\n");
    compare_rmse("IP", rmse_ip,  "Py", rmse_py,  N, 0.01f);

    /* IP vs SW */
    printf("\n  IP vs SW:\n");
    {
        float max_diff=0,mean_diff=0; int mm=0;
        for(int i=0;i<N;i++){
            float d=fabsf(rmse_ip[i]-rmse_c[i]);
            if(d>0.01f)mm++;
            if(d>max_diff)max_diff=d;
            mean_diff+=d;
        }
        mean_diff/=N;
        printf("  IP vs SW:  maxDiff=%.4e  meanDiff=%.4e  mm(>0.01)=%d/%d  %s\n",
               max_diff,mean_diff,mm,N,mm==0?"OK":"DIFF");
    }

    /* ── AUC ─────────────────────────────────────────────────────────────── */
    printf("\n══════════════════════════════════════════════════\n");
    printf("  AUC\n");
    printf("══════════════════════════════════════════════════\n");
    long n_pos=0; for(int i=0;i<N;i++) if(gt_mask[i]) n_pos++;
    if(!n_pos){
        printf("  GT mask empty — skipped\n");
    } else {
        printf("  GT mask: %ld anomaly / %d total\n",n_pos,N);
        float auc_py=compute_auc(rmse_py,gt_mask,N);
        float auc_sw=compute_auc(rmse_c, gt_mask,N);
        float auc_ip=compute_auc(rmse_ip,gt_mask,N);
        printf("  AUC  Py=%.6f  SW=%.6f  IP=%.6f\n",auc_py,auc_sw,auc_ip);
        printf("  IP-Py=%.6f  IP-SW=%.6f  SW-Py=%.6f\n",
               fabsf(auc_ip-auc_py),fabsf(auc_ip-auc_sw),fabsf(auc_sw-auc_py));
    }

    /* ── Final verdict ───────────────────────────────────────────────────── */
    printf("\n══════════════════════════════════════════════════\n");
    printf("  FINAL VERDICT\n");
    printf("══════════════════════════════════════════════════\n");

    int sw_py_mm=0, ip_py_mm=0, ip_sw_mm=0;
    for(int i=0;i<N;i++){
        if(fabsf(rmse_c[i] -rmse_py[i])>0.01f) sw_py_mm++;
        if(fabsf(rmse_ip[i]-rmse_py[i])>0.01f) ip_py_mm++;
        if(fabsf(rmse_ip[i]-rmse_c[i]) >0.01f) ip_sw_mm++;
    }
    printf("  Stage mm (SW vs Py) : %d  %s\n",stage_mm, stage_mm==0?"PASS":"FAIL");
    printf("  RMSE  mm (SW vs Py) : %d  %s\n",sw_py_mm, sw_py_mm==0?"PASS":"FAIL");
    printf("  RMSE  mm (IP vs Py) : %d  %s\n",ip_py_mm, ip_py_mm==0?"PASS":"FAIL");
    printf("  RMSE  mm (IP vs SW) : %d  %s\n",ip_sw_mm, ip_sw_mm==0?"PASS":"FAIL");
    int all_ok=(stage_mm==0&&sw_py_mm==0&&ip_py_mm==0&&ip_sw_mm==0);
    printf("\n  %s\n", all_ok
           ? "ALL PASS  —  IP core matches SW and Python T3"
           : "DIFFERENCES FOUND  —  check above");

    free(rmse_c); free(rmse_ip);
    free(image_raw); free(gt_mask); free(rmse_py);
    free_model(&model);
    return all_ok ? 0 : 1;
}