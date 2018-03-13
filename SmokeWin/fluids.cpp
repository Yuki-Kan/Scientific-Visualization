// Usage: Drag with the mouse to add smoke to the fluid. This will also move a "rotor" that disturbs
//        the velocity field at the mouse location. Press the indicated keys to change options
//--------------------------------------------------------------------------------------------------


#include "rfftw.h"              //the numerical simulation FFTW library
#include <stdio.h>              //for printing the help text
#include <math.h>               //for various math functions
#include <GL/glut.h>            //the GLUT graphics library
#include <GL/glui.h>
#include <string>
#include "transform.h"
#include <algorithm>

//--- SIMULATION PARAMETERS ------------------------------------------------------------------------
const float M_PI = 3.14159;
const int DIM = 100;			//size of simulation grid
double dt = 0.4;				//simulation time step
float visc = 0.001;				//fluid viscosity
fftw_real *vx, *vy;             //(vx,vy)   = velocity field at the current moment
fftw_real *vx0, *vy0;           //(vx0,vy0) = velocity field at the previous moment
fftw_real *fx, *fy;	            //(fx,fy)   = user-controlled simulation forces, steered with the mouse
fftw_real *rho, *rho0;			//smoke density at the current (rho) and previous (rho0) moment
rfftwnd_plan plan_rc, plan_cr;  //simulation domain discretization


//--- VISUALIZATION PARAMETERS ---------------------------------------------------------------------
int   winWidth, winHeight;      //size of the graphics window, in pixels
int   color_dir = 0;            //use direction color-coding or not
float vec_scale = 1000;			//scaling of hedgehogs
int   draw_smoke = 0;           //draw the smoke or not
int   draw_vecs = 1;            //draw the vector field or not
const int COLOR_BLACKWHITE=0;   //different types of color mapping: black-and-white, rainbow, banded
const int COLOR_RAINBOW=1;
const int COLOR_BANDS=2;
int   scalar_col = 0;           //method for scalar coloring
const int RHO=0;   //different datasets of color mapping: rho, velocity, force
const int VELO=1;
const int FORCE=2;
int scalr_data = 0;
int   frozen = 0;               //toggles on/off the animation
int   NLEVELS = 2;


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
float saturation = 1;

float col[6][3] ={{1,0,0},  // red
	                      {0,1,0},  // green
	                      {0,0,1},  // blue
	                      {1,1,0},  // yellow
	                      {0,1,1},  // cyan
	                      {1,0,1}}; // purple;
const int hh = 20;
//int bot[6][2] = {{0,0}, {50,0}, {50,0}, {100,0}, {100,0}, {250,0}},
//	       top[6][2] = {{0,hh}, {50,hh}, {50,hh}, {100,hh}, {100,hh}, {250,hh}};


//------ SIMULATION CODE STARTS HERE -----------------------------------------------------------------

// init_simulation: Initialize simulation data structures as a function of the grid size 'n'.
//                 Although the simulation takes place on a 2D grid, we allocate all data structures as 1D arrays,
//                 for compatibility with the FFTW numerical library.
void init_simulation(int n)
{
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
void FFT(int direction,void* vx)
{
	if(direction==1) rfftwnd_one_real_to_complex(plan_rc,(fftw_real*)vx,(fftw_complex*)vx);
	else             rfftwnd_one_complex_to_real(plan_cr,(fftw_complex*)vx,(fftw_real*)vx);
}

int clamp(float x)
{ return ((x)>=0.0?((int)(x)):(-((int)(1-(x))))); }

float max(float x, float y)
{ return x > y ? x : y; }

//solve: Solve (compute) one step of the fluid flow simulation
void solve(int n, fftw_real* vx, fftw_real* vy, fftw_real* vx0, fftw_real* vy0, fftw_real visc, fftw_real dt)
{
	fftw_real x, y, x0, y0, f, r, U[2], V[2], s, t;
	int i, j, i0, j0, i1, j1;

	for (i=0;i<n*n;i++)
	{ vx[i] += dt*vx0[i]; 
          vx0[i] = vx[i]; 
          vy[i] += dt*vy0[i]; 
          vy0[i] = vy[i]; }

	for ( x=0.5f/n,i=0 ; i<n ; i++,x+=1.0f/n )
	   for ( y=0.5f/n,j=0 ; j<n ; j++,y+=1.0f/n )
	   {
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

	for (i=0;i<=n;i+=2)
	{
	   x = 0.5f*i;
	   for (j=0;j<n;j++)
	   {
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
	   for (j=0;j<n;j++)
	   {
           vx[i+n*j] = f*vx0[i+(n+2)*j];
           vy[i+n*j] = f*vy0[i+(n+2)*j];
           
           float v_magnitude;
           v_magnitude = sqrt(vx[i+n*j] * vx[i+n*j] + vy[i+n*j] * vy[i+n*j]);
           
           if (v_magnitude > v_max){
               v_max = v_magnitude;
           }
           else if (v_magnitude <= v_min){
               v_min = v_magnitude;
           }
       }
}


// diffuse_matter: This function diffuses matter that has been placed in the velocity field. It's almost identical to the
// velocity diffusion step in the function above. The input matter densities are in rho0 and the result is written into rho.
void diffuse_matter(int n, fftw_real *vx, fftw_real *vy, fftw_real *rho, fftw_real *rho0, fftw_real dt)
{
	fftw_real x, y, x0, y0, s, t;
	int i, j, i0, j0, i1, j1;

	for ( x=0.5f/n,i=0 ; i<n ; i++,x+=1.0f/n )
		for ( y=0.5f/n,j=0 ; j<n ; j++,y+=1.0f/n )
		{
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
            }
            else if (rho_value <= rho_min){
                rho_min = rho_value;
            }
        
        }
}

//set_forces: copy user-controlled forces to the force vectors that are sent to the solver.
//            Also dampen forces and matter density to get a stable simulation.
void set_forces(void)
{
	int i;
	for (i = 0; i < DIM * DIM; i++)
	{
        rho0[i]  = 0.995 * rho[i];
        fx[i] *= 0.85;
        fy[i] *= 0.85;
        vx0[i]    = fx[i];
        vy0[i]    = fy[i];
        float f_magnitude;
        f_magnitude = sqrt(fx[i] * fx[i] + fy[i] * fy[i]);
        if (f_magnitude > f_max){
            f_max = f_magnitude;
        }
        else if (f_magnitude <= f_min){
            f_min = f_magnitude;
        }
	}
}


//do_one_simulation_step: Do one complete cycle of the simulation:
//      - set_forces:
//      - solve:            read forces from the user
//      - diffuse_matter:   compute a new set of velocities
//      - gluPostRedisplay: draw a new visualization frame
void do_one_simulation_step(void)
{
	if (!frozen)
	{
	  set_forces();
	  solve(DIM, vx, vy, vx0, vy0, visc, dt);
	  diffuse_matter(DIM, vx, vy, rho, rho0, dt);
	  glutPostRedisplay();
	}
}

void do_one_simulation_step2(void)
{
	if (!frozen)
	{
	  glutPostRedisplay();
	}
}


//------ VISUALIZATION CODE STARTS HERE -----------------------------------------------------------------

float test = 0;

//rainbow: Implements a color palette, mapping the scalar 'value' to a rainbow color RGB
void heatmap(float s_value,float* R,float* G,float* B)
{   
//   const float dx=0.8f;
   s_value = (s_value < s_min)? s_min : ( s_value > s_max)? s_max : s_value;
//   s_value = (6-2*dx)*s_value+dx;
    
    *R = max(0.0f, -((s_value-0.9)*(s_value-0.9))+1);
    *G = max(0.0f, -((s_value-1.5)*(s_value-1.5))+1);// max(0.0f, -(value-1)*-(value-1)+1);
    *B = 0;
}


void self_rainbow(float s_value,float* R,float* G,float* B)
{
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
//    float limitMax;
//    float limitMin;
    float baseMax;
    float baseMin;
    if(scalr_data == RHO){
//        limitMax = rho_max;
//        limitMin = rho_min;
        baseMax = clamp_rho_max;
        baseMin = clamp_rho_min;
    }
    else if (scalr_data== VELO){
//        limitMax = v_max;
//        limitMin = v_min;
        baseMax = clamp_v_max;
        baseMin = clamp_v_min;
    }else if (scalr_data== FORCE){
//        limitMax = f_max;
//        limitMin = f_min;
        baseMax = clamp_f_max;
        baseMin = clamp_f_min;
    }
    valueIn = clamp_v(valueIn, baseMax, baseMin);
    return (valueIn - baseMin) / (baseMax - baseMin);
}

// ======= hue and saturation
Rgb TransformHS(const Rgb &in, float H, float S)
{
    if(H==0)H=1;
    float SU = S*cos(H*M_PI/180);
    float SW = S*sin(H*M_PI/180);
    
    Rgb ret;
    ret.r = (.299+.701*SU+.168*SW)*in.r
    + (.587-.587*SU+.330*SW)*in.g
    + (.114-.114*SU-.497*SW)*in.b;
    ret.g = (.299-.299*SU-.328*SW)*in.r
    + (.587+.413*SU+.035*SW)*in.g
    + (.114-.114*SU+.292*SW)*in.b;
    ret.b = (.299-.3*SU+1.25*SW)*in.r
    + (.587-.588*SU-1.05*SW)*in.g
    + (.114+.886*SU-.203*SW)*in.b;
    return ret;
}


//set_colormap: Sets three different types of colormaps
void set_colormap(float v_value)
{
   float R,G,B;
    v_value = scale(v_value);
   
   if (scalar_col==COLOR_BLACKWHITE) //0
   {
       R = G = B = v_value;
   }
   else if (scalar_col==COLOR_RAINBOW) //1
   {
       v_value *= NLEVELS; v_value = (int)(v_value); v_value/= NLEVELS;
       heatmap(v_value,&R,&G,&B);
   }
   else if (scalar_col==COLOR_BANDS) //2
    {

        v_value *= NLEVELS; v_value = (int)(v_value); v_value/= NLEVELS;
        self_rainbow(v_value,&R,&G,&B);
    }

    Rgb color = {R,G,B};
    Rgb new_color = TransformHS(color, hue, saturation);
    
    glColor3f(new_color.r,new_color.g,new_color.b);
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

void draw_color_bar(){
    int i;
    float band_w = 400;
    
    float unit_length = band_w/(NLEVELS + 1);
    float unit_color = (s_max - s_min)/(NLEVELS);
    // Use Quad strips to make color bar.
    glBegin (GL_QUAD_STRIP);
     if (scalar_col==COLOR_BLACKWHITE)
        {
            float temp_color[3] = {0.0f, 0.0f, 0.0f};
            glColor3fv (temp_color);
            int unit_left[2] = {winWidth-hh, 0};
            glVertex2iv (unit_left);
            int unit_right[2] = {winWidth, 0};
            glVertex2iv (unit_right);
            
            float temp_color2[3] = {s_max, s_max, s_max};
            glColor3fv  (temp_color2);
            int unit_left2[2] = {winWidth-hh, (int)band_w};
            glVertex2iv (unit_left2);
            int unit_right2[2] = {winWidth, (int)band_w};
            glVertex2iv (unit_right2);
        }else{
            for (i = 0; i <= NLEVELS; i++)  {
               //float temp_color[3] = {start[0] + i*unit_color[0], start[1] + i*unit_color[1], start[2] + i*unit_color[2]};
                float temp_color[3] = {calculate_color_R(s_min + i*unit_color), calculate_color_G(s_min + i*unit_color), calculate_color_B(s_min + i*unit_color)};

               if(scalar_col==COLOR_RAINBOW)
            {
                float s_value = s_min + i*unit_color;
                temp_color[0] = max(0.0f, -((s_value-0.9)*(s_value-0.9))+1);
                temp_color[1] = max(0.0f, -((s_value-1.5)*(s_value-1.5))+1);
                temp_color[2] = 0.0f;
            }

              else if(scalar_col==COLOR_BANDS)
            {
               temp_color[0] = calculate_color_R(s_min + i*unit_color);
               temp_color[1] = calculate_color_G(s_min + i*unit_color);
               temp_color[2] = calculate_color_B(s_min + i*unit_color);

            }

            Rgb color = {temp_color[0], temp_color[1], temp_color[2]};
            Rgb new_color = TransformHS(color, hue, saturation);
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


void set_color_bar(){
    draw_color_bar();
}

void Color_Bar(void)
{
    if (draw_smoke){
        printf("---");
        set_color_bar();
    }else{
    int window_w = 250;
    int window_h = 10;
 
    // Set up coordinate system to position color bar near bottom of window.
 
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    glOrtho (0.0f, window_w, window_h, 0.0f, 0.0f, 1.0f);
    glMatrixMode (GL_MODELVIEW);
    glLoadIdentity ();
 
    // Use Quad strips to make color bar.
    set_color_bar();
    // Label ends of color bar.
 
    glColor3f (1, 1, 1);
    }
//    bitmap_output (-5, 7, 0, "Min_H", GLUT_BITMAP_9_BY_15);
//    bitmap_output (95, 7, 0, "Max_H", GLUT_BITMAP_9_BY_15);
}


//visualize: This is the main visualization function
void visualize(void)
{
	int        i, j, idx, idx0, idx1, idx2, idx3; double px0,py0,px1,py1,px2,py2,px3,py3;

        std::string smin,smin2,smin3,smax ;
        
	fftw_real  wn = (fftw_real)winWidth / (fftw_real)(DIM + 1);   // Grid cell width
	fftw_real  hn = (fftw_real)winHeight / (fftw_real)(DIM + 1);  // Grid cell heigh

	if (draw_smoke)
	{
            
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glBegin(GL_TRIANGLES);
	for (j = 0; j < DIM - 1; j++)			//draw smoke
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
            set_colormap(rho[idx0]);    glVertex2f(px0, py0);
            set_colormap(rho[idx1]);    glVertex2f(px1, py1);
            set_colormap(rho[idx2]);    glVertex2f(px2, py2);
            
            set_colormap(rho[idx0]);    glVertex2f(px0, py0);
            set_colormap(rho[idx2]);    glVertex2f(px2, py2);
            set_colormap(rho[idx3]);    glVertex2f(px3, py3);
            
		}
                
	}
	glEnd();
	}


	float ng = 100;
	int size = (int)((float)DIM / sqrt(ng));
	int sizeX = (int)((float)winHeight / sqrt(ng));	
	int sizeY = (int)((float)winWidth / sqrt(ng));
	if (draw_vecs)
	{
	  glBegin(GL_LINES);				//draw velocities
	  for (i = 0; i < DIM; i++)
	    for (j = 0; j < DIM; j++)
	    {
		  idx = (j * DIM) + i;
            float val_mag = 0;
			
			if (scalr_data == VELO) {
				//                direction_to_color(vx[idx],vy[idx],color_dir);
				val_mag = sqrt(vx[idx] * vx[idx] + vy[idx] * vy[idx]);
				float arrow_len = val_mag / (clamp_v_max - clamp_v_min);
				float axis_len = arrow_len / 2;
				float arrow1 = (wn + (fftw_real)i * wn) - axis_len * vx[idx] * vec_scale*sizeX;
				float arrow2 = (hn + (fftw_real)j * hn) - axis_len * vy[idx] * vec_scale*sizeY;
				float arrow3 = (wn + (fftw_real)i * wn) + axis_len * vx[idx] * vec_scale*sizeX;
				float arrow4 = (hn + (fftw_real)j * hn) + axis_len * vy[idx] * vec_scale*sizeY;

				float diff_x = arrow3 - arrow1;
				float diff_y = arrow4 - arrow2;
				if (abs(diff_x) > sizeX / 2) {
					diff_x = diff_x * ((sizeX / 2) / abs(diff_x));
					float new_mid = (arrow3 + arrow1) / 2;
					arrow3 = new_mid + diff_x / 2;
					arrow1 = new_mid - diff_x / 2;
				}
				if (abs(diff_y) > sizeY / 2) {
					diff_y = diff_y * ((sizeY / 2) / abs(diff_y));
					float new_mid = (arrow4 + arrow2) / 2;
					arrow4 = new_mid + diff_y / 2;
					arrow2 = new_mid - diff_y / 2;
				}

				if (i%size == 0 && j%size == 0) {
					glVertex2f(arrow1, arrow2);
					glVertex2f(arrow3, arrow4);
				}

            }else if (scalr_data== FORCE){
//                direction_to_color(fx[idx],fy[idx],color_dir);
                val_mag = sqrt(fx[idx] * fx[idx] + fy[idx] * fy[idx]);
                glVertex2f(wn + (fftw_real)i * wn, hn + (fftw_real)j * hn);
                glVertex2f((wn + (fftw_real)i * wn) + vec_scale * fx[idx], (hn + (fftw_real)j * hn) + vec_scale * fy[idx]);
            }
            
            set_colormap(val_mag);
	    }
	  glEnd();
	}
    
    set_color_bar();
    
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


//------ INTERACTION CODE STARTS HERE -----------------------------------------------------------------



//display: Handle window redrawing events. Simply delegates to visualize().
void display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	visualize();
//        Color_Bar();
	glFlush();
	glutSwapBuffers();
}

//display: Handle window redrawing events. Simply delegates to visualize().
void display2(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
        Color_Bar();
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
	  case 'x': draw_smoke = 1 - draw_smoke;
		    if (draw_smoke==0) draw_vecs = 1; break;
	  case 'y': draw_vecs = 1 - draw_vecs;
		    if (draw_vecs==0) draw_smoke = 1; break;
	  case 'm': scalar_col++; if (scalar_col>COLOR_BANDS) scalar_col=COLOR_BLACKWHITE; break;
	  case 'n': scalr_data++; if (scalr_data>FORCE) scalr_data=RHO;
            if (scalr_data == RHO){
                draw_smoke = 1;
                draw_vecs = 0;
            }else{
                draw_smoke = 0;
                draw_vecs = 1;
            }
            break;
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
            
        case '7': hue -=0.1;  if (hue <= 0) hue = 0; break;
        case '8': hue +=0.1;  if (hue >= 1) hue = 1; break;
        case '9': saturation -=0.1;  if (saturation <= 0) saturation = 0; break;
        case '0': saturation +=0.1;  if (saturation >= 1.5) saturation = 1.5; printf(std::to_string(saturation).c_str()); break;
	}
}



// drag: When the user drags with the mouse, add a force that corresponds to the direction of the mouse
//       cursor movement. Also inject some new matter into the field at the mouse location.
void drag(int mx, int my)
{
	int xi,yi,X,Y; double  dx, dy, len;
	static int lmx=0,lmy=0;				//remembers last mouse location

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
//        glutInitWindowPosition(100, 100);
	glutInitWindowSize(400,400);
	glutCreateWindow("Real-time smoke simulation and visualization");
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutIdleFunc(do_one_simulation_step);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(drag);
	init_simulation(DIM);	//initialize the simulation data structures
        
        
//        glutInitWindowSize(300,50);
//        glutCreateWindow("value");
//        glutDisplayFunc(display2);
//	glutIdleFunc(do_one_simulation_step2);
        
        
	glutMainLoop();			//calls do_one_simulation_step, keyboard, display, drag, reshape
	return 0;
}
