// Usage: Drag with the mouse to add smoke to the fluid. This will also move a "rotor" that disturbs
//        the velocity field at the mouse location. Press the indicated keys to change options
//--------------------------------------------------------------------------------------------------


#include "rfftw.h"              //the numerical simulation FFTW library
#include <stdio.h>              //for printing the help text
#include <math.h>               //for various math functions
#include <GLUT/glut.h>            //the GLUT graphics library
#include <GLUI/glui.h>
#include <string>
#include "transform.h"
#include <algorithm>

//--- SIMULATION PARAMETERS ------------------------------------------------------------------------
const int DIM = 100;            //size of simulation grid
double dt = 0.4;                //simulation time step
float visc = 0.001;                //fluid viscosity
fftw_real *vx, *vy;             //(vx,vy)   = velocity field at the current moment
fftw_real *vx0, *vy0;           //(vx0,vy0) = velocity field at the previous moment
fftw_real *fx, *fy;                //(fx,fy)   = user-controlled simulation forces, steered with the mouse
fftw_real *rho, *rho0;            //smoke density at the current (rho) and previous (rho0) moment
rfftwnd_plan plan_rc, plan_cr;  //simulation domain discretization


//--- VISUALIZATION PARAMETERS ---------------------------------------------------------------------
int   winWidth, winHeight;      //size of the graphics window, in pixels
int   color_dir = 0;            //use direction color-coding or not
float vec_scale = 1000;            //scaling of hedgehogs
int   draw_smoke = 0;           //draw the smoke or not
int   draw_vecs = 0;            //draw the vector field or not
int draw_gradient=0;
int draw_streamline=0;
int draw_slice=0;
const int COLOR_BLACKWHITE=0;   //different types of color mapping: black-and-white, rainbow, banded
const int COLOR_RAINBOW=1;
const int COLOR_BANDS=2;
int   scalar_col = 0;           //method for scalar coloring
int vector_col = 0;
int gradient_map = 0;
const int VELO=0;   //different datasets of color mapping: rho, velocity, force
const int FORCE=1;
const int RHO=2;
int scalr_data = 0;
int vect_data = 0;
int   frozen = 0;               //toggles on/off the animation
int   NLEVELS = 2;
int changeHS=0;
int is_scale=0;
int is_reset=0;


//---customize parameters
float s_min = 0;
float s_max = 1;
float rho_min = 0;
float rho_max = 0;
float v_min = 0;
float v_max = 0;
float f_min = 0;
float f_max = 0;

float clamp_rho_min = 0;
float clamp_rho_max = 0;
float clamp_v_min = 0;
float clamp_v_max = 0;
float clamp_f_min = 0;
float clamp_f_max = 0;

float hue = 0;
float saturation = 0;

float col[6][3] ={{1,0,0},  // red
                          {0,1,0},  // green
                          {0,0,1},  // blue
                          {1,1,0},  // yellow
                          {0,1,1},  // cyan
                          {1,0,1}}; // purple;
const int hh = 20;

int main_window;
int number_of_glyphs = 100;
std::string glyphs = "hedgehogs";
int glyph = 0;
int hedge = 0;
int triangle = 1;
int arrow = 2;
float arrow_scale = 100;
fftw_real  cell_width = ceil((fftw_real)winWidth / (fftw_real)(DIM));   // Grid cell width
fftw_real  cell_height = ceil((fftw_real)winHeight / (fftw_real)(DIM));  // Grid cell heigh
fftw_real  wn = (fftw_real)(winWidth-20) / (fftw_real)(DIM + 1);   // Grid cell width
fftw_real  hn = (fftw_real)(winHeight-20) / (fftw_real)(DIM + 1);  // Grid cell heigh
int gradient_col = 0;   //0: gradient_smoke  1:gradient_velocity
float gradient_size_smoke = 5;
float gradient_size_velo = 5;
float mouse_px;
float mouse_py;
bool has_click = false;


//------ SIMULATION CODE STARTS HERE -----------------------------------------------------------------

// init_simulation: Initialize simulation data structures as a function of the grid size 'n'.
//                 Although the simulation takes place on a 2D grid, we allocate all data structures as 1D arrays,
//                 for compatibility with the FFTW numerical library.
void init_simulation(int n){
    int i; size_t dim;

    dim     = n * 2*(n/2+1)*sizeof(fftw_real);        //Allocate data structures
    vx       = (fftw_real*) malloc(dim);
    vy       = (fftw_real*) malloc(dim);
    vx0      = (fftw_real*) malloc(dim);
    vy0      = (fftw_real*) malloc(dim);
    dim     = n * n * sizeof(fftw_real);
    fx      = (fftw_real*) malloc(dim);
    fy      = (fftw_real*) malloc(dim);
    rho     = (fftw_real*) malloc(dim);
    rho0    = (fftw_real*) malloc(dim);
    plan_rc = rfftw2d_create_plan(n, n, FFTW_REAL_TO_COMPLEX, FFTW_IN_PLACE);
    plan_cr = rfftw2d_create_plan(n, n, FFTW_COMPLEX_TO_REAL, FFTW_IN_PLACE);

    for (i = 0; i < n * n; i++)                      //Initialize data structures to 0
    { vx[i] = vy[i] = vx0[i] = vy0[i] = fx[i] = fy[i] = rho[i] = rho0[i] = 0.0f; }
}


//FFT: Execute the Fast Fourier Transform on the dataset 'vx'.
//     'dirfection' indicates if we do the direct (1) or inverse (-1) Fourier Transform
void FFT(int direction,void* vx){
    if(direction==1) rfftwnd_one_real_to_complex(plan_rc,(fftw_real*)vx,(fftw_complex*)vx);
    else             rfftwnd_one_complex_to_real(plan_cr,(fftw_complex*)vx,(fftw_real*)vx);
}

int clamp(float x)
{ return ((x)>=0.0?((int)(x)):(-((int)(1-(x))))); }

float max(float x, float y)
{ return x > y ? x : y; }

//solve: Solve (compute) one step of the fluid flow simulation
void solve(int n, fftw_real* vx, fftw_real* vy, fftw_real* vx0, fftw_real* vy0, fftw_real visc, fftw_real dt){
    fftw_real x, y, x0, y0, f, r, U[2], V[2], s, t;
    int i, j, i0, j0, i1, j1;

    for (i=0;i<n*n;i++){
        vx[i] += dt*vx0[i];
        vx0[i] = vx[i];
        vy[i] += dt*vy0[i];
        vy0[i] = vy[i];
    }

    for ( x=0.5f/n,i=0 ; i<n ; i++,x+=1.0f/n )
       for ( y=0.5f/n,j=0 ; j<n ; j++,y+=1.0f/n ){
          x0 = n*(x-dt*vx0[i+n*j])-0.5f;
          y0 = n*(y-dt*vy0[i+n*j])-0.5f;
          i0 = clamp(x0); s = x0-i0;
          i0 = (n+(i0%n))%n;
          i1 = (i0+1)%n;
          j0 = clamp(y0); t = y0-j0;
          j0 = (n+(j0%n))%n;
          j1 = (j0+1)%n;
          vx[i+n*j] = (1-s)*((1-t)*vx0[i0+n*j0]+t*vx0[i0+n*j1])+s*((1-t)*vx0[i1+n*j0]+t*vx0[i1+n*j1]);
          vy[i+n*j] = (1-s)*((1-t)*vy0[i0+n*j0]+t*vy0[i0+n*j1])+s*((1-t)*vy0[i1+n*j0]+t*vy0[i1+n*j1]);
       }

    for(i=0; i<n; i++)
      for(j=0; j<n; j++)
      {  vx0[i+(n+2)*j] = vx[i+n*j]; vy0[i+(n+2)*j] = vy[i+n*j]; }

    FFT(1,vx0);
    FFT(1,vy0);

    for (i=0;i<=n;i+=2){
       x = 0.5f*i;
       for (j=0;j<n;j++){
          y = j<=n/2 ? (fftw_real)j : (fftw_real)j-n;
          r = x*x+y*y;
          if ( r==0.0f ) continue;
          f = (fftw_real)exp(-r*dt*visc);
          U[0] = vx0[i  +(n+2)*j]; V[0] = vy0[i  +(n+2)*j];
          U[1] = vx0[i+1+(n+2)*j]; V[1] = vy0[i+1+(n+2)*j];

          vx0[i  +(n+2)*j] = f*((1-x*x/r)*U[0]     -x*y/r *V[0]);
          vx0[i+1+(n+2)*j] = f*((1-x*x/r)*U[1]     -x*y/r *V[1]);
          vy0[i+  (n+2)*j] = f*(  -y*x/r *U[0] + (1-y*y/r)*V[0]);
          vy0[i+1+(n+2)*j] = f*(  -y*x/r *U[1] + (1-y*y/r)*V[1]);
       }
    }

    FFT(-1,vx0);
    FFT(-1,vy0);

    f = 1.0/(n*n);
     for (i=0;i<n;i++)
       for (j=0;j<n;j++){
           vx[i+n*j] = f*vx0[i+(n+2)*j];
           vy[i+n*j] = f*vy0[i+(n+2)*j];
           
           float v_magnitude;
           v_magnitude = 100*sqrt(vx[i+n*j] * vx[i+n*j] + vy[i+n*j] * vy[i+n*j]);
           if (v_magnitude > v_max){
               v_max = v_magnitude;
           }else if (v_magnitude <= v_min){
               v_min = v_magnitude;
           }
       }
}


// diffuse_matter: This function diffuses matter that has been placed in the velocity field. It's almost identical to the
// velocity diffusion step in the function above. The input matter densities are in rho0 and the result is written into rho.
void diffuse_matter(int n, fftw_real *vx, fftw_real *vy, fftw_real *rho, fftw_real *rho0, fftw_real dt){
    fftw_real x, y, x0, y0, s, t;
    int i, j, i0, j0, i1, j1;

    for ( x=0.5f/n,i=0 ; i<n ; i++,x+=1.0f/n )
        for ( y=0.5f/n,j=0 ; j<n ; j++,y+=1.0f/n ){
            x0 = n*(x-dt*vx[i+n*j])-0.5f;
            y0 = n*(y-dt*vy[i+n*j])-0.5f;
            i0 = clamp(x0);
            s = x0-i0;
            i0 = (n+(i0%n))%n;
            i1 = (i0+1)%n;
            j0 = clamp(y0);
            t = y0-j0;
            j0 = (n+(j0%n))%n;
            j1 = (j0+1)%n;
            rho[i+n*j] = (1-s)*((1-t)*rho0[i0+n*j0]+t*rho0[i0+n*j1])+s*((1-t)*rho0[i1+n*j0]+t*rho0[i1+n*j1]);
            float rho_value = rho[i+n*j];
            if (rho_value > rho_max){
                rho_max = rho_value;
            }else if (rho_value <= rho_min){
                rho_min = rho_value;
            }
        }
}

//set_forces: copy user-controlled forces to the force vectors that are sent to the solver.
//            Also dampen forces and matter density to get a stable simulation.
void set_forces(void){
    int i;
    for (i = 0; i < DIM * DIM; i++){
        rho0[i]  = 0.995 * rho[i];
        fx[i] *= 0.85;
        fy[i] *= 0.85;
        vx0[i] = fx[i];
        vy0[i] = fy[i];
        float f_magnitude;
        f_magnitude = 100 * sqrt(fx[i] * fx[i] + fy[i] * fy[i]);
        if (f_magnitude > f_max){
            f_max = f_magnitude;
        }else if (f_magnitude <= f_min){
            f_min = f_magnitude;
        }
    }
}


//do_one_simulation_step: Do one complete cycle of the simulation:
//      - set_forces:
//      - solve:            read forces from the user
//      - diffuse_matter:   compute a new set of velocities
//      - gluPostRedisplay: draw a new visualization frame
void do_one_simulation_step(void){
    if (!frozen){
      set_forces();
      solve(DIM, vx, vy, vx0, vy0, visc, dt);
      diffuse_matter(DIM, vx, vy, rho, rho0, dt);
      glutPostRedisplay();
    }
}

void do_one_simulation_step2(void){
    if (!frozen){
      glutPostRedisplay();
    }
}


//------ VISUALIZATION CODE STARTS HERE -----------------------------------------------------------------

//rainbow: Implements a color palette, mapping the scalar 'value' to a rainbow color RGB
void heatmap(float s_value,float* R,float* G,float* B){
    s_value = (s_value < s_min)? s_min : ( s_value > s_max)? s_max : s_value;
    *R = max(0.0f, -((s_value-0.9)*(s_value-0.9))+1);
    *G = max(0.0f, -((s_value-1.5)*(s_value-1.5))+1);
    *B = 0;
}

void self_rainbow(float s_value,float* R,float* G,float* B){
   const float dx=0.8f;
   s_value = (s_value < s_min)? s_min : ( s_value > s_max)? s_max : s_value;    //clamp scalar value in [min, max]
   s_value = (6-2*dx)*s_value+dx;
   
   *R = max(0.0f,(3-(float)fabs(s_value-4)-(float)fabs(s_value-5))/2);
   *G = max(0.0f,(4-(float)fabs(s_value-2)-(float)fabs(s_value-4))/2);
   *B = max(0.0f,(3-(float)fabs(s_value-1)-(float)fabs(s_value-2))/2);
}

float clamp_v(float v_value, float clamp_max, float clamp_min){
    if (v_value > clamp_max){
        v_value = clamp_max;
    }else if (v_value < clamp_min){
        v_value = clamp_min;
    }
    return v_value;
}

float scale(float valueIn){
    float baseMax;
    float baseMin;
    if(scalr_data == RHO){
        baseMax = clamp_rho_max;
        baseMin = clamp_rho_min;
    }
    else if (scalr_data == VELO){
        baseMax = clamp_v_max;
        baseMin = clamp_v_min;
    }else if (scalr_data == FORCE){
        baseMax = clamp_f_max;
        baseMin = clamp_f_min;
    }
    valueIn = clamp_v(valueIn, baseMax, baseMin);
    return (valueIn - baseMin) / (baseMax - baseMin);
}

// ======= hue and saturation ====
Hsl rgb2hsv(Rgb ret){
    Hsl hsv;
    float M = fmax(ret.r, fmax(ret.g,ret.b));
    float m = fmin(ret.r, fmin(ret.g,ret.b));
    float d = M-m;
    
    hsv.l = M;
    hsv.s = (M>0.00001)? d/M: 0 ; //saturation
    if (hsv.s==0) hsv.h=0 ; //achromatic case , hue=0 by convention
    else{  //chromatic case
        if (ret.r==M) hsv.h = (ret.g-ret.b)/d ;
        else if (ret.g==M) hsv.h = 2 + (ret.b-ret.r )/d;
        else hsv.h = 4 + ( ret.r-ret.g)/d ;
        hsv.h/= 6;
        if (hsv.h<0) hsv.h+=1 ;
    }
    return hsv;
}

Rgb hsv2rgb(Hsl hsv, float H, float S){
    Rgb ret;
    hsv.h += H;
    hsv.s += S;
    if (hsv.h > 1){
        hsv.h = hsv.h - 1;
    }
    if (hsv.s > 1){
        hsv.s = hsv.s - 1;
    }
    
    int hueCase = (int)(hsv.h*6);
    float frac = hsv.h*6-hueCase;
    float lx = hsv.l*(1-hsv.s);
    float ly = hsv.l*(1-hsv.s*frac);
    float lz = hsv.l*(1-hsv.s*(1-frac));
    
    switch (hueCase){
        case 0:
        case 6: ret.r=hsv.l ; ret.g=lz ; ret.b=lx ; break ; // 0<hue<1/6
        case 1: ret.r=ly ; ret.g=hsv.l ; ret.b=lx ; break ; // 1/6<hue<2/6
        case 2: ret.r=lx ; ret.g=hsv.l ; ret.b=lz ; break ; // 2/6<hue<3/6
        case 3: ret.r=lx ; ret.g=ly ; ret.b=hsv.l ; break ; // 3/6<hue/4/6
        case 4: ret.r=lz ; ret.g=lx ; ret.b=hsv.l ; break ; // 4/6<hue<5/6
        case 5: ret.r=hsv.l ; ret.g=lx ; ret.b=ly ; break ; // 5/6<hue<1
    }
    return ret;
}


//set_colormap: Sets three different types of colormaps
void set_colormap(float v_value){
   float R,G,B;
    if (is_scale){
        v_value = scale(v_value);
    }
   if (scalar_col==COLOR_BLACKWHITE){
       R = G = B = v_value;
   }else if (scalar_col==COLOR_RAINBOW){
       v_value *= NLEVELS; v_value = (int)(v_value); v_value/= NLEVELS;
       heatmap(v_value,&R,&G,&B);
   }else if (scalar_col==COLOR_BANDS){
        v_value *= NLEVELS; v_value = (int)(v_value); v_value/= NLEVELS;
        self_rainbow(v_value,&R,&G,&B);
    }

    Rgb color = {R,G,B};
    if(changeHS == 1){
            Hsl new_hsv = rgb2hsv(color);
            Rgb new_color = hsv2rgb(new_hsv, hue, saturation);
            glColor3f(new_color.r,new_color.g,new_color.b);
    }else{
        glColor3f(color.r, color.g, color.b);
    }
}


//set_colormap_vector: Sets three different types of colormaps
void set_colormap_vector(float v_value){
    float R,G,B;
    if (is_scale){
        v_value = scale(v_value);
    }
    
    if (vector_col==COLOR_BLACKWHITE){
        R = G = B = v_value;
    }else if (vector_col==COLOR_RAINBOW){
        v_value *= NLEVELS; v_value = (int)(v_value); v_value/= NLEVELS;
        heatmap(v_value,&R,&G,&B);
    }else if (vector_col==COLOR_BANDS){
        v_value *= NLEVELS; v_value = (int)(v_value); v_value/= NLEVELS;
        self_rainbow(v_value,&R,&G,&B);
    }
    
    Rgb color = {R,G,B};
    if(changeHS == 1){
        Hsl new_hsv = rgb2hsv(color);
        Rgb new_color = hsv2rgb(new_hsv, hue, saturation);
        glColor3f(new_color.r,new_color.g,new_color.b);
    }else{
        glColor3f(color.r, color.g, color.b);
    }
}

//set_colormap_vector: Sets three different types of colormaps
void set_colormap_gradient(float v_value){
    float R,G,B;
    if (is_scale){
        v_value = scale(v_value);
    }
    
    if (gradient_map==COLOR_BLACKWHITE){
        R = G = B = v_value;
    }else if (gradient_map==COLOR_RAINBOW){
        v_value *= NLEVELS; v_value = (int)(v_value); v_value/= NLEVELS;
        heatmap(v_value,&R,&G,&B);
    }else if (gradient_map==COLOR_BANDS){
        v_value *= NLEVELS; v_value = (int)(v_value); v_value/= NLEVELS;
        self_rainbow(v_value,&R,&G,&B);
    }
    
    Rgb color = {R,G,B};
    if(changeHS == 1){
        Hsl new_hsv = rgb2hsv(color);
        Rgb new_color = hsv2rgb(new_hsv, hue, saturation);
        glColor3f(new_color.r,new_color.g,new_color.b);
    }else{
        glColor3f(color.r, color.g, color.b);
    }
}


//direction_to_color: Set the current color by mapping a direction vector (x,y), using
//                    the color mapping method 'method'. If method==1, map the vector direction
//                    using a rainbow colormap. If method==0, simply use the white color
void direction_to_color(float x, float y, int method)
{
    float r,g,b,f;
    if (method)
    {
      f = atan2(y,x) / 3.1415927 + 1;
      r = f;
      if(r > 1) r = 2 - r;
      g = f + .66667;
      if(g > 2) g -= 2;
      if(g > 1) g = 2 - g;
      b = f + 2 * .66667;
      if(b > 2) b -= 2;
      if(b > 1) b = 2 - b;
    }
    else
    { r = g = b = 1; }
    glColor3f(r,g,b);
}


void displayText( float x, float y, int r, int g, int b, const char *string ) {
    int j = strlen( string );
    glColor3f( r, g, b );
    glRasterPos2f( x, y );
    for( int i = 0; i < j; i++ ) {
        glutBitmapCharacter( GLUT_BITMAP_TIMES_ROMAN_24, string[i] );
    }
}



//--------------------------------  Color_Bar  ---------------------------------


float calculate_color_R(float val){
    float dx = 0.8f;
    float value_ = (6-2*dx)* val + dx;
    return max(0.0f,(3-(float)fabs(value_-4)-(float)fabs(value_-5))/2);
}

float calculate_color_G(float val){
    float dx = 0.8f;
    float value_ = (6-2*dx)* val + dx;
    return max(0.0f,(4-(float)fabs(value_-2)-(float)fabs(value_-4))/2);
}

float calculate_color_B(float val){
    float dx = 0.8f;
    float value_ = (6-2*dx)* val + dx;
    return max(0.0f,(3-(float)fabs(value_-1)-(float)fabs(value_-2))/2);
}


void draw_color_bar_vector(){
    int i;
    int barlen = 500;
    float unit_length = barlen/(NLEVELS + 1);
    float unit_color = (s_max - s_min)/(NLEVELS);
    // Use Quad strips to make color bar.
    glBegin (GL_QUAD_STRIP);
    if (vector_col==COLOR_BLACKWHITE){
        float temp_color[3] = {0.0f, 0.0f, 0.0f};
        glColor3fv (temp_color);
        int unit_left[2] = {winWidth-2*hh-barlen, 0};
        glVertex2iv (unit_left);
        int unit_right[2] = {winWidth-2*hh-barlen, hh};
        glVertex2iv (unit_right);
        
        float temp_color2[3] = {s_max, s_max, s_max};
        glColor3fv  (temp_color2);
        int unit_left2[2] = {winWidth-2*hh,0};
        glVertex2iv (unit_left2);
        int unit_right2[2] = {winWidth-2*hh, hh};
        glVertex2iv (unit_right2);
    }else{
        for (i = 0; i <= NLEVELS; i++) {
            float temp_color[3] = {calculate_color_R(s_min + i*unit_color), calculate_color_G(s_min + i*unit_color), calculate_color_B(s_min + i*unit_color)};
            
            if(vector_col==COLOR_RAINBOW){
                float s_value = s_min + i*unit_color;
                temp_color[0] = max(0.0f, -((s_value-0.9)*(s_value-0.9))+1);
                temp_color[1] = max(0.0f, -((s_value-1.5)*(s_value-1.5))+1);
                temp_color[2] = 0.0f;
            }else if(vector_col==COLOR_BANDS){
                temp_color[0] = calculate_color_R(s_min + i*unit_color);
                temp_color[1] = calculate_color_G(s_min + i*unit_color);
                temp_color[2] = calculate_color_B(s_min + i*unit_color);
            }
            
            Rgb color = {temp_color[0], temp_color[1], temp_color[2]};
            Rgb new_color = color;
            if(changeHS == 1){
                Hsl new_hsv = rgb2hsv(color);
                new_color = hsv2rgb(new_hsv, hue, saturation);
            }else{
                new_color = color;
            }
            temp_color[0] = new_color.r;
            temp_color[1] = new_color.g;
            temp_color[2] = new_color.b;
            
            glColor3fv  (temp_color);
            int unit_left[2] = {winWidth-2*hh-barlen+(int)(i*unit_length), 0};
            glVertex2iv (unit_left);
            int unit_right[2] = {winWidth-2*hh-barlen+(int)(i*unit_length), hh};
            glVertex2iv (unit_right);
            
            glColor3fv  (temp_color);
            int unit2_left[2] = {winWidth-2*hh-barlen+(int)((i+1)*unit_length), 0};
            glVertex2iv (unit2_left);
            int unit2_right[2] = {winWidth-2*hh-barlen+(int)((i+1)*unit_length), hh};
            glVertex2iv (unit2_right);
        }
    }
    glEnd ();
}


void draw_color_bar_scalar(){
    int i;
    float unit_length = winWidth/(NLEVELS + 1);
    float unit_color = (s_max - s_min)/(NLEVELS);
    // Use Quad strips to make color bar.
    glBegin (GL_QUAD_STRIP);
    if (scalar_col==COLOR_BLACKWHITE){
        float temp_color[3] = {0.0f, 0.0f, 0.0f};
        glColor3fv (temp_color);
        int unit_left[2] = {winWidth-hh, 0};
        glVertex2iv (unit_left);
        int unit_right[2] = {winWidth, 0};
        glVertex2iv (unit_right);
        
        float temp_color2[3] = {s_max, s_max, s_max};
        glColor3fv  (temp_color2);
        int unit_left2[2] = {winWidth-hh, (int)winWidth};
        glVertex2iv (unit_left2);
        int unit_right2[2] = {winWidth, (int)winWidth};
        glVertex2iv (unit_right2);
    }else{
        for (i = 0; i <= NLEVELS; i++)  {
            float temp_color[3] = {calculate_color_R(s_min + i*unit_color), calculate_color_G(s_min + i*unit_color), calculate_color_B(s_min + i*unit_color)};
            
            if(scalar_col==COLOR_RAINBOW){
                float s_value = s_min + i*unit_color;
                temp_color[0] = max(0.0f, -((s_value-0.9)*(s_value-0.9))+1);
                temp_color[1] = max(0.0f, -((s_value-1.5)*(s_value-1.5))+1);
                temp_color[2] = 0.0f;
            }else if(scalar_col==COLOR_BANDS){
                temp_color[0] = calculate_color_R(s_min + i*unit_color);
                temp_color[1] = calculate_color_G(s_min + i*unit_color);
                temp_color[2] = calculate_color_B(s_min + i*unit_color);
            }
            
            Rgb color = {temp_color[0], temp_color[1], temp_color[2]};
            Rgb new_color = color;
            if(changeHS == 1){
                Hsl new_hsv = rgb2hsv(color);
                new_color = hsv2rgb(new_hsv, hue, saturation);
            }else{
                new_color = color;
            }
            temp_color[0] = new_color.r;
            temp_color[1] = new_color.g;
            temp_color[2] = new_color.b;
            
            glColor3fv  (temp_color);
            int unit_left[2] = {winWidth-hh, (int)(i*unit_length)};
            glVertex2iv (unit_left);
            int unit_right[2] = {winWidth, (int)(i*unit_length)};
            glVertex2iv (unit_right);
            
            glColor3fv  (temp_color);
            int unit2_left[2] = {winWidth - hh, (int)((i+1)*unit_length)};
            glVertex2iv (unit2_left);
            int unit2_right[2] = {winWidth, (int)((i+1)*unit_length)};
            glVertex2iv (unit2_right);
        }
    }
    glEnd ();
}

void dispBarValScala(){
    std::string smin,smin2,smin3,smax ;
    float draw_max = 0.5;
    float draw_min = 0.0;
    if(scalr_data == RHO){
        draw_max = clamp_rho_max;
        draw_min = clamp_rho_min;
    }else if (scalr_data== VELO){
        draw_max = clamp_v_max;
        draw_min = clamp_v_min;
    }else if (scalr_data == FORCE){
        draw_max = clamp_f_max;
        draw_min = clamp_f_min;
    }
    
    float interval = (draw_max-draw_min)/3;
    smin = std::to_string(draw_min);
    smin2 = std::to_string(draw_min + interval);
    smin3 = std::to_string(draw_min + interval*2);
    smax = std::to_string(draw_max);
    
    displayText(winWidth - 2*hh,(winHeight-hh),1,0,0, smax.c_str());
    displayText(winWidth - 2*hh,(winHeight-hh)*2/3,1,0,0, smin3.c_str());
    displayText(winWidth - 2*hh,(winHeight-hh)*1/3,1,0,0, smin2.c_str());
    displayText(winWidth - 2*hh,0,1,0,0, smin.c_str());
}

void dispBarValVector(){
    int barlen = 500;
    std::string smin,smin2,smin3,smax ;
    float draw_max = 0.5;
    float draw_min = 0.0;
    if(vect_data == RHO){
        draw_max = clamp_rho_max;
        draw_min = clamp_rho_min;
    }else if (vect_data== VELO){
        draw_max = clamp_v_max;
        draw_min = clamp_v_min;
    }else if (vect_data == FORCE){
        draw_max = clamp_f_max;
        draw_min = clamp_f_min;
    }
    
    float interval = (draw_max-draw_min)/3;
    smin = std::to_string(draw_min);
    smin2 = std::to_string(draw_min + interval);
    smin3 = std::to_string(draw_min + interval*2);
    smax = std::to_string(draw_max);
    
    displayText(winWidth-2*hh,1.5*hh,1,0,0, smax.c_str());
    displayText(winWidth-2*hh-barlen*1/3,1.5*hh,1,0,0, smin3.c_str());
    displayText(winWidth-2*hh-barlen*2/3,1.5*hh,1,0,0, smin2.c_str());
    displayText(winWidth-2*hh-barlen,1.5*hh,1,0,0, smin.c_str());
}

//visualize: This is the main visualization function

float angle2DVector(float vec_vx, float vec_vy){
    float vec_len = sqrt(vec_vx * vec_vx + vec_vy * vec_vy);
    float norm_vx = vec_vx/vec_len;
    float norm_vy = vec_vy/vec_len;
    float angle = atan2 (norm_vx,-norm_vy) * (180 / M_PI);
    return angle;
}

float len2DVector(float vec_vx, float vec_vy){
    float vec_len = sqrt(vec_vx * vec_vx + vec_vy * vec_vy);
    return vec_len;
}

float len3DVector(float vec_vx, float vec_vy, float z){
    float vec_len = sqrt(vec_vx * vec_vx + vec_vy * vec_vy + z*z);
    return vec_len;
}


void drawArrow(float vx1, float vx2, float vy1, float vy2,float vy){
    // draw an arrow the size of a cell, scale according to vector length
    float vec_vx = vx1 - vx2;
    float vec_vy = vy1 - vy2;
    float angle = angle2DVector(vec_vx, vec_vy);
    
    set_colormap_vector(vy);
    glPushMatrix();
    glTranslatef(vx1,vy1, 0);
    glRotated(angle,0,0,1);

    float len = len2DVector(vec_vx, vec_vy);
    GLUquadricObj *quadObj;
    int D = 5;
    
    glTranslatef(0,0,len-4*D);

    quadObj = gluNewQuadric ();
    gluQuadricDrawStyle (quadObj, GLU_FILL);
    gluQuadricNormals (quadObj, GLU_SMOOTH);
    gluCylinder(quadObj, 2*D, 0.0, 4*D, 32, 1);
    gluDeleteQuadric(quadObj);
    
    quadObj = gluNewQuadric ();
    gluQuadricDrawStyle (quadObj, GLU_FILL);
    gluQuadricNormals (quadObj, GLU_SMOOTH);
    gluDisk(quadObj, 0.0, 2*D, 32, 1);
    gluDeleteQuadric(quadObj);
    
    glTranslatef(0,0,-len+4*D);
    
    quadObj = gluNewQuadric ();
    gluQuadricDrawStyle (quadObj, GLU_FILL);
    gluQuadricNormals (quadObj, GLU_SMOOTH);
    gluCylinder(quadObj, D, D, len-4*D, 32, 1);
    gluDeleteQuadric(quadObj);
    
    quadObj = gluNewQuadric ();
    gluQuadricDrawStyle (quadObj, GLU_FILL);
    gluQuadricNormals (quadObj, GLU_SMOOTH);
    gluDisk(quadObj, 0.0, D, 32, 1);
    gluDeleteQuadric(quadObj);
    
    glPopMatrix ();
    glLoadIdentity(); // needed to stop the rotating, otherwise rotates the entire drawing
}

#define RADPERDEG 0.0174533
void draw3D(float vx1, float vx2, float vy1, float vy2, float vy){
    float x = vx1 - vx2;
    float y = vy1 - vy2;
    float z = 8;
    float len = len3DVector(x, y, z);
    
    set_colormap_vector(vy);
    glPushMatrix();
    glTranslatef(vx1,vy1,2);
    
    if((x!=0.)||(y!=0.)) {
        glRotated(atan2(y,x)/RADPERDEG,0.,0.,1.);
        glRotated(atan2(sqrt(x*x+y*y),z)/RADPERDEG,0.,-1.,0.);
    } else if (z<0){
        glRotated(180,1.,0.,0.);
    }
    
//    float angle = angle2DVector(vx1, vx2, vy1, vy2);
//    glRotated(angle,0,0,1);
    
    GLUquadricObj *quadObj;
    int D = 5;
    
    glTranslatef(0,0,len-4*D);
    
    quadObj = gluNewQuadric ();
    gluQuadricDrawStyle (quadObj, GLU_FILL);
    gluQuadricNormals (quadObj, GLU_SMOOTH);
    gluCylinder(quadObj, 2*D, 0.0, 4*len, 16, 1);
    gluDeleteQuadric(quadObj);
    
    quadObj = gluNewQuadric ();
    gluQuadricDrawStyle (quadObj, GLU_FILL);
    gluQuadricNormals (quadObj, GLU_SMOOTH);
    gluDisk(quadObj, 0.0, 2*D, 16, 1);
    gluDeleteQuadric(quadObj);
    
    glTranslatef(0,0,-len+4*D);

    quadObj = gluNewQuadric ();
    gluQuadricDrawStyle (quadObj, GLU_FILL);
    gluQuadricNormals (quadObj, GLU_SMOOTH);
    gluCylinder(quadObj, D, 0, len-4*len, 16, 1);
    gluDeleteQuadric(quadObj);

    quadObj = gluNewQuadric ();
    gluQuadricDrawStyle (quadObj, GLU_FILL);
    gluQuadricNormals (quadObj, GLU_SMOOTH);
    gluDisk(quadObj, 0.0, D, 16, 1);
    gluDeleteQuadric(quadObj);
    
    glPopMatrix ();
    glLoadIdentity(); // needed to stop the rotating, otherwise rotates the entire drawing
    
}


void drawAxes(float vx1, float vx2, float vy1, float vy2,float vy)
{
//    float len = len2DVector(vx1, vx2, vy1, vy2);
    float len = 10;
    glPushMatrix();
    glTranslatef(len,0,0);
//    drawArrow(vx1, vx2, vy1, vy2, vy);
    draw3D(vx1, vx2, vy1, vy2, vy);
    glPopMatrix();
    
    glPushMatrix();
    glTranslatef(0,len,0);
//    drawArrow(vx1, vx2, vy1, vy2, vy);
    draw3D(vx1, vx2, vy1, vy2, vy);
    glPopMatrix();
    
    glPushMatrix();
    glTranslatef(0,0,len);
//    drawArrow(vx1, vx2, vy1, vy2, vy);
    draw3D(vx1, vx2, vy1, vy2, vy);
    glPopMatrix();
}


void stepStreamLine(float px, float py, int step, int current_step){
    int i, j, idx, idx0, idx1, idx2, idx3;
    float px0,py0,px1,py1,px2,py2,px3,py3;
    float dist = 0.5 *wn;
    bool drawing_flag = true;
    glBegin(GL_LINES);
    while (drawing_flag && current_step <= step) {
        for (i = 0; i < DIM; i++) {
            for (j = 0; j < DIM; j++) {
                px0 = wn + (fftw_real)i * wn;
                py0 = hn + (fftw_real)j * hn;
                idx0 = (j * DIM) + i;
                
                px1 = wn + (fftw_real)i * wn;
                py1 = hn + (fftw_real)(j + 1) * hn;
                idx1 = ((j + 1) * DIM) + i;
                
                px2 = wn + (fftw_real)(i + 1) * wn;
                py2 = hn + (fftw_real)(j + 1) * hn;
                idx2 = ((j + 1) * DIM) + (i + 1);
                
                px3 = wn + (fftw_real)(i + 1) * wn;
                py3 = hn + (fftw_real)j * hn;
                idx3 = (j * DIM) + (i + 1);
                
                if (px > px0 && px < px2 && py > py0 && py < py2) {
                    float x_diff = px - px0;
                    float y_diff = py - py0;
                    
                    float vx_bot = (vx[idx3] - vx[idx0])*x_diff / wn + vx[idx0];
                    float vx_top = (vx[idx2] - vx[idx1])*x_diff / wn + vx[idx1];
                    float px_v = (vx_top - vx_bot)*y_diff / hn + vx_bot;
                    
                    float vy_bot = (vy[idx3] - vy[idx0])*x_diff / wn + vy[idx0];
                    float vy_top = (vy[idx2] - vy[idx1])*x_diff / wn + vy[idx1];
                    float py_v = (vy_top - vy_bot)*y_diff / hn + vy_bot;
                    
                    float p0_length = len2DVector(px_v, py_v);
                    
                    float dt = dist / p0_length;
                    float pnext_x = px + px_v * dt;
                    float pnext_y = py + py_v * dt;
                    
                    if (pnext_x < winWidth && pnext_x > 0 && pnext_y < winHeight && pnext_y > 0) {
                        if (dt < 3000) {
                            glBegin(GL_LINES);
                            glVertex2f(px, py);
                            glVertex2f(pnext_x, pnext_y);
                            set_colormap_vector(100 * p0_length);
                            px = pnext_x;
                            py = pnext_y;
                        }
                        else {
                            drawing_flag = false;
                            break;
                        }
                    }
                    
                }
            }
        }
        current_step = current_step + 1;
    }
    glEnd();
}

void visualize(void){
    int i, j, idx, idx0, idx1, idx2, idx3; double px0,py0,px1,py1,px2,py2,px3,py3;

    fftw_real  wn = (fftw_real)(winWidth-20) / (fftw_real)(DIM + 1);   // Grid cell width
    fftw_real  hn = (fftw_real)(winHeight-20) / (fftw_real)(DIM + 1);  // Grid cell heigh

    if (draw_smoke){
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glBegin(GL_TRIANGLES);
    for (j = 0; j < DIM - 1; j++)            //draw smoke
    {
        for (i = 0; i < DIM - 1; i++)
        {
            px0  = wn + (fftw_real)i * wn;
            py0  = hn + (fftw_real)j * hn;
            idx0 = (j * DIM) + i;

            px1  = wn + (fftw_real)i * wn;
            py1  = hn + (fftw_real)(j + 1) * hn;
            idx1 = ((j + 1) * DIM) + i;

            px2  = wn + (fftw_real)(i + 1) * wn;
            py2  = hn + (fftw_real)(j + 1) * hn;
            idx2 = ((j + 1) * DIM) + (i + 1);

            px3  = wn + (fftw_real)(i + 1) * wn;
            py3  = hn + (fftw_real)j * hn;
            idx3 = (j * DIM) + (i + 1);
            
            if(scalr_data == RHO){
                set_colormap(rho[idx0]);    glVertex2f(px0, py0);
                set_colormap(rho[idx1]);    glVertex2f(px1, py1);
                set_colormap(rho[idx2]);    glVertex2f(px2, py2);
                
                set_colormap(rho[idx0]);    glVertex2f(px0, py0);
                set_colormap(rho[idx2]);    glVertex2f(px2, py2);
                set_colormap(rho[idx3]);    glVertex2f(px3, py3);
            }
            else if (scalr_data == VELO){
                float vel0 = 100*sqrt(vx[idx0]*vx[idx0]+vy[idx0]*vy[idx0]);
                float vel1 = 100*sqrt(vx[idx1]*vx[idx1]+vy[idx1]*vy[idx1]);
                float vel2 = 100*sqrt(vx[idx2]*vx[idx2]+vy[idx2]*vy[idx2]);
                float vel3 = 100*sqrt(vx[idx3]*vx[idx3]+vy[idx3]*vy[idx3]);
                
                set_colormap(vel0);    glVertex2f(px0, py0);
                set_colormap(vel1);    glVertex2f(px1, py1);
                set_colormap(vel2);    glVertex2f(px2, py2);
                
                set_colormap(vel0);    glVertex2f(px0, py0);
                set_colormap(vel2);    glVertex2f(px2, py2);
                set_colormap(vel3);    glVertex2f(px3, py3);
            }
            else if (scalr_data == FORCE){
                float force0 = 100*sqrt(fx[idx0]*fx[idx0]+fy[idx0]*fy[idx0]);
                float force1 = 100*sqrt(fx[idx1]*fx[idx1]+fy[idx1]*fy[idx1]);
                float force2 = 100*sqrt(fx[idx2]*fx[idx2]+fy[idx2]*fy[idx2]);
                float force3 = 100*sqrt(fx[idx3]*fx[idx3]+fy[idx3]*fy[idx3]);
                
                set_colormap(force0);    glVertex2f(px0, py0);
                set_colormap(force1);    glVertex2f(px1, py1);
                set_colormap(force2);    glVertex2f(px2, py2);
                
                set_colormap(force0);    glVertex2f(px0, py0);
                set_colormap(force2);    glVertex2f(px2, py2);
                set_colormap(force3);    glVertex2f(px3, py3);
            }
        }
    }
        glEnd();
        draw_color_bar_scalar();
        dispBarValScala();
    }

    if (draw_vecs){
      glBegin(GL_LINES);                //draw velocities
      for (i = 0; i < DIM; i++)
        for (j = 0; j < DIM; j++){
          idx = (j * DIM) + i;
          float val_mag = 0;
            
          if (vect_data== VELO){
                if (glyph == hedge) {
                    if (j % 5==0 && i % 5==0){
                        val_mag = sqrt(vx[idx] * vx[idx] + vy[idx] * vy[idx]);
                        set_colormap_vector(100*val_mag);
                        glVertex2f(wn + (fftw_real)i * wn, hn + (fftw_real)j * hn);
                        glVertex2f((wn + (fftw_real)i * wn) + vec_scale * vx[idx], (hn + (fftw_real)j * hn) + vec_scale * vy[idx]);
                    }
            }else if (glyph == arrow){
                    if (j % 2==0 && i % 2==0){
                        float x1 = wn+(fftw_real)i * wn;
                        float y1 = hn+(fftw_real)j * hn;
                        float x2 = (wn + (fftw_real)i * wn) + vec_scale * vx[idx];
                        float y2 = (hn + (fftw_real)j * hn) + vec_scale * vy[idx];
                        float vec_vx = x1 - x2;
                        float vec_vy = y1 - y2;
                        float len = len2DVector(vec_vx, vec_vy);
//                        drawArrow(x1, x2, y1, y2, len/15);
//                        drawAxes(x1, x2, y1, y2, len/15);
                        draw3D(x1, x2, y1, y2, len/10);
                        
                    }
            }else if (glyph == triangle){
                    if (j % 4==0 && i % 4==0){
                        val_mag = sqrt(vx[idx] * vx[idx] + vy[idx] * vy[idx]);
                        glBegin(GL_TRIANGLE_STRIP);
                        set_colormap_vector(100*val_mag);
                        int scale = 300;
                        glVertex2f((fftw_real)i*hn + scale * vy[idx], (fftw_real)j*wn - scale * vx[idx]);
                        glVertex2f((fftw_real)i*hn - scale * vy[idx], (fftw_real)j*wn + scale * vx[idx]);
                        glVertex2f((fftw_real)i*wn + vec_scale * vx[idx], (fftw_real)j*hn + vec_scale * vy[idx]);
                        glEnd();
                        glFlush();
                    }
                }
                else if (glyphs == "3D"){
                    
                    
                }
        }else if (vect_data== FORCE){
                if (glyph == hedge) {
                        val_mag = sqrt(fx[idx] * fx[idx] + fy[idx] * fy[idx]);
                        set_colormap_vector(100*val_mag);
                        glVertex2f(wn + (fftw_real)i * wn, hn + (fftw_real)j * hn);
                        glVertex2f((wn + (fftw_real)i * wn) + vec_scale * fx[idx], (hn + (fftw_real)j * hn) + vec_scale * fy[idx]);
                
                }else if (glyph == arrow){
                    if (j % 5==0 && i % 5==0){
                        float x1 = wn+(fftw_real)i * wn;
                        float y1 = hn+(fftw_real)j * hn;
                        float x2 = (wn + (fftw_real)i * wn) + vec_scale * fx[idx];
                        float y2 = (hn + (fftw_real)j * hn) + vec_scale * fy[idx];
                        float vec_vx = x1 - x2;
                        float vec_vy = y1 - y2;
                        float len = len2DVector(vec_vx, vec_vy);
                        drawArrow(x1, x2, y1, y2, len/15);
//                        drawAxes(x1, x2, y1, y2, len/15);
                    }
                }else if (glyph == triangle){
                    glBegin(GL_TRIANGLE_STRIP);
                    val_mag = sqrt(fx[idx] * fx[idx] + fy[idx] * fy[idx]);
                    set_colormap_vector(100*val_mag);
                    int scale = 300;
                    glVertex2f((fftw_real)i*hn + scale * fy[idx], (fftw_real)j*wn - scale * fx[idx]);
                    glVertex2f((fftw_real)i*hn - scale * fy[idx], (fftw_real)j*wn + scale * fx[idx]);
                    glVertex2f((fftw_real)i*wn + vec_scale * fx[idx], (fftw_real)j*hn + vec_scale * fy[idx]);
                    glEnd();
                    glFlush();
                }
                else if (glyphs == "3D"){
                    
                }
                
            }
        }
        glEnd();
        draw_color_bar_vector();
        dispBarValVector();
    }
    
    
// ============= gradient ====================
    
    if(draw_gradient){
        float px0,py0,px1,py1,px2,py2,px3,py3;
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//        glBegin(GL_LINES);
//            glBegin(GL_TRIANGLE_STRIP);
            for (i = 0; i < DIM; i++)
                for (j = 0; j < DIM; j++){
                    idx = (j * DIM) + i;
                    px0  = wn + (fftw_real)i * wn;
                    py0  = hn + (fftw_real)j * hn;
                    idx0 = (j * DIM) + i;
                    
                    px1  = wn + (fftw_real)i * wn;
                    py1  = hn + (fftw_real)(j + 1) * hn;
                    idx1 = ((j + 1) * DIM) + i;
                    
                    px2  = wn + (fftw_real)(i + 1) * wn;
                    py2  = hn + (fftw_real)(j + 1) * hn;
                    idx2 = ((j + 1) * DIM) + (i + 1);
                    
                    px3  = wn + (fftw_real)(i + 1) * wn;
                    py3  = hn + (fftw_real)j * hn;
                    idx3 = (j * DIM) + (i + 1);
                    
                    if(gradient_col==0){     //density
                        if (j % 2==0 && i % 2==0){
                            float px = 0.5*(px0+px2);
                            float py = 0.5*(py0+py2);
                            float dfx = 0.5*(rho[idx1]-rho[idx0])/wn + 0.5*(rho[idx2]-rho[idx3])/wn;
                            float dfy = 0.5*(rho[idx3]-rho[idx0])/hn + 0.5*(rho[idx2]-rho[idx1])/hn;
                            float length = len2DVector(dfx, dfy);
                            int scale = 48;
                            int triscale = 8;
//                            glBegin(GL_LINES);
//                            set_colormap_gradient(10*length);
//                            glVertex2f(px, py);
//                            glVertex2f(px + scale * dfx, py + scale*dfy);
//                            glBegin(GL_TRIANGLE_STRIP);
                            glBegin(GL_TRIANGLES);
                            set_colormap_gradient(100*length);
                            glVertex2f(px + scale * dfx, py + scale * dfy);
                            glVertex2f(px + triscale * dfy, py - triscale * dfx);
                            glVertex2f(px - triscale * dfy, py + triscale * dfx);
                        }
                        
                    }else if(gradient_col == 1){
                        if (j % 2==0 && i % 2==0){
                        float px = 0.5*(px2+px0);
                        float py = 0.5*(py0+py2);
                        float mag_v0 = len2DVector(vx[idx0], vy[idx0]);
                        float mag_v1 = len2DVector(vx[idx1], vy[idx1]);
                        float mag_v2 = len2DVector(vx[idx2], vy[idx2]);
                        float mag_v3 = len2DVector(vx[idx3], vy[idx3]);
                        float dfx = 0.5*(mag_v1-mag_v0)/wn + 0.5*(mag_v2-mag_v3)/wn;
                        float dfy = 0.5*(mag_v3-mag_v0)/hn + 0.5*(mag_v2-mag_v1)/hn;
                        float length = len2DVector(dfx, dfy);
                        int scale = 10000;
                        int triscale = 1000;
//                        glBegin(GL_LINES);
//                        set_colormap_gradient(1000*length);
//                        glVertex2f(px, py);
//                        glVertex2f(px + scale * dfx, py + scale*dfy);
                        glBegin(GL_TRIANGLES);
                        set_colormap_gradient(1000*length);
                        glVertex2f(px + scale * dfx, py + scale*dfy);
                        glVertex2f(px + triscale*dfy, py - triscale*dfx);
                        glVertex2f(px - triscale*dfy, py + triscale*dfx);
//                            glVertex2f((fftw_real)i*hn + scale * dfy, (fftw_real)j*wn - scale * dfx);
//                            glVertex2f((fftw_real)i*hn - scale * dfy, (fftw_real)j*wn + scale * dfx);
//                            glVertex2f((fftw_real)i*wn + triscale * dfx, (fftw_real)j*hn + triscale * dfy);
                        }
        }
                    
        }
        glEnd();
    }
   
    if(draw_streamline){
        if (has_click){
            stepStreamLine(mouse_px, mouse_py, 200, 0);
        }
    }
    
    if(draw_slice){
        //write here
    }
}


//------ INTERACTION CODE STARTS HERE -----------------------------------------------------------------

/***************************************** myGlutIdle() ***********/

void myGlutIdle( void )
{
    /* According to the GLUT specification, the current window is
     undefined during an idle callback.  So we need to explicitly change
     it if necessary */
    if ( glutGetWindow() != main_window )
        glutSetWindow(main_window);
    
    glutPostRedisplay();
}


//display: Handle window redrawing events. Simply delegates to visualize().
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    visualize();
    glFlush();
    glutSwapBuffers();
}


//reshape: Handle window resizing (reshaping) events
void reshape(int w, int h)
{
    glViewport(0.0f, 0.0f, (GLfloat)w, (GLfloat)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, (GLdouble)w, 0.0, (GLdouble)h);
    winWidth = w; winHeight = h;
    wn = (fftw_real)(winWidth-20) / (fftw_real)(DIM + 1);   // Grid cell width
    hn = (fftw_real)(winHeight-20) / (fftw_real)(DIM + 1);
    glutPostRedisplay();
}


//keyboard: Handle key presses
void keyboard(unsigned char key, int x, int y)
{
    switch (key)
    {
        case 'r': clamp_rho_max = rho_max; clamp_f_max = f_max; clamp_v_max = v_max; clamp_rho_min = rho_min;
            clamp_v_min = v_min; clamp_f_min = f_min; break;
        case 't': dt -= 0.001; break;
        case 'T': dt += 0.001; break;
        case 'c': color_dir = 1 - color_dir; break;
        case 'S': vec_scale *= 1.2; break;
        case 's': vec_scale *= 0.8; break;
        case 'V': visc *= 5; break;
        case 'v': visc *= 0.2; break;
        case 'x': draw_smoke = 1 - draw_smoke;break;  // check
            
        case 'y': draw_vecs = 1 - draw_vecs;break;  // check
        case 'o': draw_gradient = 1 - draw_gradient;break;
        case 'u': changeHS = 1-changeHS; break;
            
        case 'i': gradient_col++; if(gradient_col>1) gradient_col=0;break;
            
            
        case 'm': scalar_col++; if (scalar_col>COLOR_BANDS) scalar_col=COLOR_BLACKWHITE; break;
        case 'k': vector_col++; if (vector_col>COLOR_BANDS) vector_col=COLOR_BLACKWHITE; break;
        case 'j': vect_data++; if(vect_data>FORCE) vect_data=VELO; break;
        case 'n': scalr_data++; if (scalr_data>RHO) scalr_data=VELO;break;
        case 'p': glyph++; if (glyph>arrow) glyph=hedge;break;
        case 'a': frozen = 1-frozen; break;
        case 'q': exit(0);
            
          case '1':
        {
            if(scalr_data == RHO){
                clamp_rho_min -= 0.5; if (clamp_rho_min < rho_min) clamp_rho_min =rho_min; break;
            }
            else if (scalr_data== VELO){
                clamp_v_min -= 0.5; if (clamp_v_min < v_min) clamp_v_min = v_min; break;
            }else if (scalr_data== FORCE){
                clamp_f_min -= 0.5; if (clamp_f_min < f_min) clamp_f_min = f_min; break;
            }
        }
            
        case '2':
        {
            if(scalr_data == RHO){
                clamp_rho_min += 0.5; if (clamp_rho_min >= clamp_rho_max) clamp_rho_min = clamp_rho_max - 0.5; break;
            }
            else if (scalr_data== VELO){
                clamp_v_min += 0.5; if (clamp_v_min >= clamp_v_max) clamp_v_min = clamp_v_max - 0.5; break;
            }else if (scalr_data== FORCE){
                clamp_f_min += 0.5; if (clamp_f_min >= clamp_f_max) clamp_f_min = clamp_f_max - 0.5; break;
            }
        }
            
        case '3':
        {
            if(scalr_data == RHO){
                clamp_rho_max -= 0.5; if (clamp_rho_min >= clamp_rho_max) clamp_rho_max = clamp_rho_min + 0.5; break;
            }
            else if (scalr_data== VELO){
                clamp_v_max -= 0.5; if (clamp_v_min >= clamp_v_max) clamp_v_max = clamp_v_min + 0.5; break;
            }else if (scalr_data== FORCE){
                clamp_f_max -= 0.5; if (clamp_f_min >= clamp_f_max) clamp_f_max = clamp_f_min + 0.5; break;
            }
        }
            
        case '4':
        {
            if(scalr_data == RHO){
                clamp_rho_max += 0.5; if (rho_max < clamp_rho_max) clamp_rho_max =rho_max; break;
            }
            else if (scalr_data== VELO){
                clamp_v_max += 0.5; if (v_max < clamp_v_max) clamp_v_max = v_max; break;
            }else if (scalr_data== FORCE){
                clamp_f_max += 0.5; if (f_max < clamp_f_max) clamp_f_max = f_max; break;
            }
        }
         
          case '6': NLEVELS +=1;  if (NLEVELS >= 256) NLEVELS = 256; break;
          case '5': NLEVELS -=1;  if (NLEVELS <= 2) NLEVELS = 2; break;
            
        case '7': hue -=0.1;  if (hue <= 0) hue = 0; printf("%f\n", hue);break;
        case '8': hue +=0.1;  if (hue >= 1) hue = 1; printf("%f\n", hue);break;
        case '9': saturation -=0.1;  if (saturation <= 0) saturation = 0; break;
        case '0': saturation +=0.1;  if (saturation >= 1.5) saturation = 1.5; printf(std::to_string(saturation).c_str()); break;
//            printf("Value: %f %f %f\n", value, value2, value3);
    }
}



// drag: When the user drags with the mouse, add a force that corresponds to the direction of the mouse
//       cursor movement. Also inject some new matter into the field at the mouse location.
void drag(int mx, int my)
{
    int xi,yi,X,Y; double  dx, dy, len;
    static int lmx=0,lmy=0;                //remembers last mouse location

    // Compute the array index that corresponds to the cursor location
    xi = (int)clamp((double)(DIM + 1) * ((double)mx / (double)winWidth));
    yi = (int)clamp((double)(DIM + 1) * ((double)(winHeight - my) / (double)winHeight));

    X = xi; Y = yi;

    if (X > (DIM - 1))  X = DIM - 1; if (Y > (DIM - 1))  Y = DIM - 1;
    if (X < 0) X = 0; if (Y < 0) Y = 0;

    // Add force at the cursor location
    my = winHeight - my;
    dx = mx - lmx; dy = my - lmy;
    len = sqrt(dx * dx + dy * dy);
    if (len != 0.0) {  dx *= 0.1 / len; dy *= 0.1 / len; }
    fx[Y * DIM + X] += dx;
    fy[Y * DIM + X] += dy;
    rho[Y * DIM + X] = 10.0f;
    lmx = mx; lmy = my;
    glutPostRedisplay();
}

void OnMouseClick(int botton, int state, int x, int y){
    if(draw_streamline && botton == GLUT_LEFT_BUTTON){
        has_click = true;
        mouse_px = x;
        mouse_py = winHeight - y;
    }
}


//main: The main program
int main(int argc, char **argv)
{
    printf("Fluid Flow Simulation and Visualization\n");
    printf("=======================================\n");
    printf("Click and drag the mouse to steer the flow!\n");
    printf("T/t:   increase/decrease simulation timestep\n");
    printf("S/s:   increase/decrease hedgehog scaling\n");
    printf("c:     toggle direction coloring on/off\n");
    printf("V/v:   increase decrease fluid viscosity\n");
    printf("x:     toggle drawing matter on/off\n");
    printf("y:     toggle drawing hedgehogs on/off\n");
    printf("m:     toggle thru scalar coloring\n");
    printf("a:     toggle the animation on/off\n");
        printf("1:     decrease min\n");
        printf("2:     increase min\n");
        printf("3:     decrease max\n");
        printf("4:     increase max\n");
        printf("5:     decrease number of colors (min = 2)\n");
        printf("6:     increase number of colors (max = 256)\n");
    printf("q:     quit\n\n");

    
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glutInitWindowSize(800,800);
    main_window = glutCreateWindow("Real-time smoke simulation and visualization");
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(drag);
    glutMouseFunc(OnMouseClick);
    init_simulation(DIM);    //initialize the simulation data structures
    
    
    /****************************************/
    /*         Here's the GLUI code         */
    /****************************************/
    printf( "GLUI version: %3.2f\n", GLUI_Master.get_version());
    
    /** Pointers to the windows and some of the controls we'll create **/
    GLUI *glui;
    GLUI_RadioGroup *radio1, *radio2, *radio3, *radio4, *radio5, *radio_grad, *grad_map;
    GLUI_Panel *obj_panel, *obj_panel2, *panel2_1, *panel2_2, *obj_panel3, *obj_panel4, *panel3_1, *panel3_2;
    GLUI_Panel *panel3_3, *panel4_2, *panel4_1;
    glui = GLUI_Master.create_glui("Menu Bar");
    
// public parameters panel
    obj_panel = new GLUI_Rollout(glui, "Properties", true );
    GLUI_Spinner *spinner = new GLUI_Spinner(obj_panel, "Bands:", &NLEVELS);
    spinner->set_int_limits(3, 256);
    spinner->set_alignment(GLUI_ALIGN_LEFT);
    GLUI_Checkbox *check1 = new GLUI_Checkbox(obj_panel, "Scale", &is_scale);
    check1->set_alignment(GLUI_ALIGN_LEFT);
//    GLUI_Checkbox *check2 = new GLUI_Checkbox(obj_panel, "Reset", &is_reset);
//    check2->set_alignment(GLUI_ALIGN_CENTER);
    
    
//    glui->add_column_to_panel(obj_panel, true);
    new GLUI_Checkbox(obj_panel, "Change Hue and Saturation", &changeHS);
    GLUI_Spinner *spinner2 = new GLUI_Spinner(obj_panel, "Hue:", &hue);
    spinner2->set_float_limits(0, 1);
    spinner2->set_alignment(GLUI_ALIGN_LEFT);

    GLUI_Spinner *spinner_sat = new GLUI_Spinner(obj_panel, "Saturation:", &saturation);
    spinner_sat->set_float_limits(0, 1);
    spinner_sat->set_alignment(GLUI_ALIGN_LEFT);
    
// scalar field panel
    obj_panel2 = new GLUI_Rollout(glui, "Scalar Field Panel", true);
    new GLUI_Checkbox(obj_panel2, "Scalar Field", &draw_smoke);
    panel2_1 = new GLUI_Panel(obj_panel2, "Dataset", true);
    radio2 = new GLUI_RadioGroup(panel2_1, &scalr_data);
    new GLUI_RadioButton(radio2, "||Velocity||");
    new GLUI_RadioButton(radio2, "||Force||");
    new GLUI_RadioButton(radio2, "Density");
    
    panel2_2 = new GLUI_Panel(obj_panel2, "ColorMaps", true);
    radio3 = new GLUI_RadioGroup(panel2_2, &scalar_col);
    new GLUI_RadioButton(radio3, "Black-White");
    new GLUI_RadioButton(radio3, "Heatmap");
    new GLUI_RadioButton(radio3, "Rainbow");
    
//vector field panel
    obj_panel3 = new GLUI_Rollout(glui, "Vector Field Panel", true);
    new GLUI_Checkbox(obj_panel3, "Vector Field", &draw_vecs);
    panel3_1 = new GLUI_Panel(obj_panel3, "Dataset", true);
    radio1 = new GLUI_RadioGroup(panel3_1, &vect_data);
    new GLUI_RadioButton(radio1, "Velocity");
    new GLUI_RadioButton(radio1, "Force");
    
    panel3_2 = new GLUI_Panel(obj_panel3, "ColorMaps", true);
    radio4 = new GLUI_RadioGroup(panel3_2, &vector_col);
    new GLUI_RadioButton(radio4, "Black-White");
    new GLUI_RadioButton(radio4, "Heatmap");
    new GLUI_RadioButton(radio4, "Rainbow");
    
    panel3_3 = new GLUI_Panel(obj_panel3, "Glyphs", true);
    radio5 = new GLUI_RadioGroup(panel3_3, &glyph);
    new GLUI_RadioButton(radio5, "Hedgehogs");
    new GLUI_RadioButton(radio5, "Triangles");
    new GLUI_RadioButton(radio5, "3D Arrows");
    
// Gradient panel
    obj_panel4 = new GLUI_Rollout(glui, "Gradient Panel", false);
    new GLUI_Checkbox(obj_panel4, "Gradient", &draw_gradient);
    panel4_1 = new GLUI_Panel(obj_panel4, "Dataset", true);
    radio_grad = new GLUI_RadioGroup(panel4_1, &gradient_col);
    new GLUI_RadioButton(radio_grad, "Density");
    new GLUI_RadioButton(radio_grad, "Velocity");

    panel4_2 = new GLUI_Panel(obj_panel4, "ColorMaps", true);
    grad_map = new GLUI_RadioGroup(panel4_2, &gradient_map);
    new GLUI_RadioButton(grad_map, "Black-White");
    new GLUI_RadioButton(grad_map, "Heatmap");
    new GLUI_RadioButton(grad_map, "Rainbow");
    
    GLUI_Rollout *obj_panel5 = new GLUI_Rollout(glui, "StreamLine Panel", false);
    new GLUI_Checkbox(obj_panel5, "StreamLine", &draw_streamline);
    
    GLUI_Rollout *obj_panel6 = new GLUI_Rollout(glui, "Slice Panel", false);
    new GLUI_Checkbox(obj_panel6, "Slice", &draw_slice);
    
    glui->set_main_gfx_window( main_window );
    GLUI_Master.set_glutIdleFunc(do_one_simulation_step);
    glutMainLoop();            //calls do_one_simulation_step, keyboard, display, drag, reshape
    return 0;
}

