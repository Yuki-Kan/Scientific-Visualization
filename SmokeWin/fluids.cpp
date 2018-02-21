// Usage: Drag with the mouse to add smoke to the fluid. This will also move a "rotor" that disturbs 
//        the velocity field at the mouse location. Press the indicated keys to change options
//-------------------------------------------------------------------------------------------------- 

#include <rfftw.h>              //the numerical simulation FFTW library
#include <GL/glut.h>            //the GLUT graphics library
#include <stdio.h>              //for printing the help text
//  Include windows library in order to use the Sleep function
#include <windows.h>

//  For sqrt
#include <math.h>

//  Include GLUI, GLUT, OpenGL, and GLU libraries
#include <glui.h>

//--- SIMULATION PARAMETERS ------------------------------------------------------------------------
const int DIM = 50;				//size of simulation grid
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
int   frozen = 0;               //toggles on/off the animation
								//  variable representing the window title
char *window_title = "Real-time smoke simulation and visualization";

//  The id of the main window
GLuint main_window;

//---customize parameters
float col[6][3] = { { 1,0,0 },  // red
{ 0,1,0 },  // green
{ 0,0,1 },  // blue
{ 1,1,0 },  // yellow
{ 0,1,1 },  // cyan
{ 1,0,1 } }; // purple;
const int hh = 15;
int bot[6][2] = { { 0,0 },{ 50,0 },{ 100,0 },{ 150,0 },{ 200,0 },{ 250,0 } },
top[6][2] = { { 0,hh },{ 50,hh },{ 100,hh },{ 150,hh },{ 200,hh },{ 250,hh } };


//*************************************************************************
//  GLUI Declarations
//*************************************************************************

//  pointer to the GLUI window
GLUI * glui_window;

//  Declare live variables (related to GLUI)
int wireframe = 1;			//  Related to Wireframe Check Box
int draw = 1;				//  Related to Draw Check Box
int listbox_item_id = 12;	//  Id of the selected item in the list box
int radiogroup_item_id = 0; //  Id of the selcted radio button
float rotation_matrix[16]	//  Rotation Matrix Live Variable Array
= { 1.0, 0.0, 0.0, 0.0,
0.0, 1.0, 0.0, 0.0,
0.0, 0.0, 1.0, 0.0,
0.0, 0.0, 0.0, 1.0 };
float translate_xy[2]		//  Translation XY Live Variable
= { 0, 0 };
float translate_z = 0;		//  Translation Z Live Variable
float scale = 1;			//  Spinner Scale Live Variable

							// an array of RGB components
float color[] = { 1.0, 1.0, 1.0 };

//  Set up the GLUI window and its components
void setupGLUI();

//  Idle callack function
void idle();

//  Declare callbacks related to GLUI
void glui_callback(int arg);

//  Declare the IDs of controls generating callbacks
enum
{
	COLOR_LISTBOX = 0,
	OBJECTYPE_RADIOGROUP,
	TRANSLATION_XY,
	TRANSLATION_Z,
	ROTATION,
	SCALE_SPINNER,
	QUIT_BUTTON
};

//  The different GLUT shapes
enum GLUT_SHAPES
{
	GLUT_WIRE_CUBE = 0,
	GLUT_SOLID_CUBE,
	GLUT_WIRE_SPHERE,
	GLUT_SOLID_SPHERE,
	GLUT_WIRE_CONE,
	GLUT_SOLID_CONE,
	GLUT_WIRE_TORUS,
	GLUT_SOLID_TORUS,
	GLUT_WIRE_DODECAHEDRON,
	GLUT_SOLID_DODECAHEDRON,
	GLUT_WIRE_OCTAHEDRON,
	GLUT_SOLID_OCTAHEDRON,
	GLUT_WIRE_TETRAHEDRON,
	GLUT_SOLID_TETRAHEDRON,
	GLUT_WIRE_ICOSAHEDRON,
	GLUT_SOLID_ICOSAHEDRON,
	GLUT_WIRE_TEAPOT,
	GLUT_SOLID_TEAPOT
};


//------ SIMULATION CODE STARTS HERE -----------------------------------------------------------------

//init_simulation: Initialize simulation data structures as a function of the grid size 'n'. 
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

//solve: Solve (compute) one step of the fluid flow simulation
void solve(int n, fftw_real* vx, fftw_real* vy, fftw_real* vx0, fftw_real* vy0, fftw_real visc, fftw_real dt) 
{
	fftw_real x, y, x0, y0, f, r, U[2], V[2], s, t;
	int i, j, i0, j0, i1, j1;

	for (i=0;i<n*n;i++) 
	{ vx[i] += dt*vx0[i]; vx0[i] = vx[i]; vy[i] += dt*vy0[i]; vy0[i] = vy[i]; }    

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
	   { vx[i+n*j] = f*vx0[i+(n+2)*j]; vy[i+n*j] = f*vy0[i+(n+2)*j]; }
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


//------ VISUALIZATION CODE STARTS HERE -----------------------------------------------------------------


//rainbow: Implements a color palette, mapping the scalar 'value' to a rainbow color RGB
void rainbow(float value,float* R,float* G,float* B)
{                          
   const float dx=0.8; 
   if (value<0) value=0; if (value>1) value=1;
   value = (6-2*dx)*value+dx;
   *R = max(0.0,(3-fabs(value-4)-fabs(value-5))/2);
   *G = max(0.0,(4-fabs(value-2)-fabs(value-4))/2);
   *B = max(0.0,(3-fabs(value-1)-fabs(value-2))/2);
}

void self_rainbow(float value, float* R, float* G, float* B)
{
	const float dx = 0.8f;
	if (value<0) value = 0; if (value>1) value = 1;
	value = (6 - 2 * dx)*value + dx;

	*R = max(0.0f, (3 - (float)fabs(value - 4) - (float)fabs(value - 5)) / 2);
	*G = max(0.0f, (4 - (float)fabs(value - 2) - (float)fabs(value - 4)) / 2);
	*B = (0.0f, 0.0f);
}

//set_colormap: Sets three different types of colormaps
void set_colormap(float vy)
{
   float R,G,B; 

   if (scalar_col==COLOR_BLACKWHITE)
       R = G = B = vy;
   else if (scalar_col==COLOR_RAINBOW)
	   self_rainbow(vy,&R,&G,&B);
   else if (scalar_col==COLOR_BANDS)
       {  
          const int NLEVELS = 7;
          vy *= NLEVELS; vy = (int)(vy); vy/= NLEVELS; 
	      rainbow(vy,&R,&G,&B);   
	   }
   
   glColor3f(R,G,B);
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


void displayText(float x, float y, int r, int g, int b, const char *string) {
	int j = strlen(string);

	glColor3f(r, g, b);
	glRasterPos2f(x, y);
	for (int i = 0; i < j; i++) {
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, string[i]);
	}
}


//--------------------------------  Color_Bar  --------------------------------- 

void draw_color_bar(float temp[6][3]) {
	int i;

	// Use Quad strips to make color bar.
	glBegin(GL_QUAD_STRIP);
	for (i = 0; i <= 5; i++) {
		glColor3fv(temp[i]);
		glVertex2iv(bot[i]);
		glVertex2iv(top[i]);
	}
	glEnd();
}

void set_color_bar() {
	int i;
	if (scalar_col == COLOR_BLACKWHITE)
	{
		float temp_array[6][3] = { { 1,1,1 },  // red
		{ 0.5,0.5,0.5 },  // green     
		{ 0.5,0.5,0.5 },  // green
		{ 0.5,0.5,0.5 },  // green
		{ 0.5,0.5,0.5 },  // green
						  //                                  {0,0,1},  // blue
						  //                                  {1,1,0},  // yellow
						  //                                  {0,1,1},  // cyan
		{ 0,0,0 } }; // purple
		draw_color_bar(temp_array);
	}

	else if (scalar_col == COLOR_RAINBOW)
	{
		// self-rainbow: R and G
		float temp_array[6][3] = { { 1,0,0 },  // red
		{ 0.5,0.5,0 },  // green
		{ 0.5,0.5,0 },  // blue
		{ 0.5,0.5,0 },  // green
		{ 0.5,0.5,0 },  // cyan
		{ 0,1,0 } }; // purple
		draw_color_bar(temp_array);
	}

	else if (scalar_col == COLOR_BANDS)
	{
		float temp_array[6][3] = { { 1,0,0 },  // red
		{ 0,1,0 },  // green
		{ 0,0,1 },  // blue
		{ 1,1,0 },  // yellow
		{ 0,1,1 },  // cyan
		{ 1,0,1 } }; // purple
		draw_color_bar(temp_array);

	}
}

void Color_Bar(void)
{
	if (draw_smoke) {
		printf("---");
		set_color_bar();
	}
	else {
		int window_w = 250;
		int window_h = 10;

		// Set up coordinate system to position color bar near bottom of window.

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(0.0f, window_w, window_h, 0.0f, 0.0f, 1.0f);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		// Use Quad strips to make color bar.
		set_color_bar();

		// Label ends of color bar.

		glColor3f(1, 1, 1);
	}
	//    bitmap_output (-5, 7, 0, "Min_H", GLUT_BITMAP_9_BY_15);
	//    bitmap_output (95, 7, 0, "Max_H", GLUT_BITMAP_9_BY_15);
}

//visualize: This is the main visualization function
void visualize(void)
{
	int        i, j, idx;
	fftw_real  wn = (fftw_real)winWidth / (fftw_real)(DIM + 1);   // Grid cell width
	fftw_real  hn = (fftw_real)winHeight / (fftw_real)(DIM + 1);  // Grid cell heigh

	if (draw_smoke)
	{	
		int idx0, idx1, idx2, idx3;
		double px0, py0, px1, py1, px2, py2, px3, py3;
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glBegin(GL_TRIANGLES);
		for (j = 0; j < DIM - 1; j++)            //draw smoke
		{
			for (i = 0; i < DIM - 1; i++)
			{
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


				set_colormap(rho[idx0]);    glVertex2f(px0, py0);
				set_colormap(rho[idx1]);    glVertex2f(px1, py1);
				set_colormap(rho[idx2]);    glVertex2f(px2, py2);


				set_colormap(rho[idx0]);    glVertex2f(px0, py0);
				set_colormap(rho[idx2]);    glVertex2f(px2, py2);
				set_colormap(rho[idx3]);    glVertex2f(px3, py3);
			}
		}
		glEnd();
		set_color_bar();
		displayText(0, 0, 1, 0, 0, "20");
	}

	if (draw_vecs)
	{
		glBegin(GL_LINES);				//draw velocities
		for (i = 0; i < DIM; i++)
			for (j = 0; j < DIM; j++)
			{
				idx = (j * DIM) + i;
				direction_to_color(vx[idx],vy[idx],color_dir);
				glVertex2f(wn + (fftw_real)i * wn, hn + (fftw_real)j * hn);
				glVertex2f((wn + (fftw_real)i * wn) + vec_scale * vx[idx], (hn + (fftw_real)j * hn) + vec_scale * vy[idx]);
			}
		glEnd();
	}
}


//------ INTERACTION CODE STARTS HERE -----------------------------------------------------------------

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
	  case 't': dt -= 0.001; break;
	  case 'T': dt += 0.001; break;
	  case 'c': color_dir = 1 - color_dir; break;
	  case 'S': vec_scale *= 1.2; break;
	  case 's': vec_scale *= 0.8; break;
	  case 'V': visc *= 5; break;
	  case 'vy': visc *= 0.2; break;
	  case 'x': draw_smoke = 1 - draw_smoke; 
		    if (draw_smoke==0) draw_vecs = 1; break;
	  case 'y': draw_vecs = 1 - draw_vecs; 
		    if (draw_vecs==0) draw_smoke = 1; break;
	  case 'm': scalar_col++; if (scalar_col>COLOR_BANDS) scalar_col=COLOR_BLACKWHITE; break;
	  case 'a': frozen = 1-frozen; break;
	  case 'q': exit(0);
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

//*************************************************************************
//  GLUI Functions.
//*************************************************************************

//-------------------------------------------------------------------------
//  Setup GLUI stuff.
//-------------------------------------------------------------------------
void setupGLUI()
{
	int window_x = 400;
	int window_y = 400;

	//  Set idle function
	GLUI_Master.set_glutIdleFunc(idle);

	//  Create GLUI window
	glui_window = GLUI_Master.create_glui("Options", 0, window_x - 235, window_y);

	//---------------------------------------------------------------------
	// 'Object Properties' Panel
	//---------------------------------------------------------------------

	//  Add the 'Object Properties' Panel to the GLUI window
	GLUI_Panel *op_panel = glui_window->add_panel("Object Properties");

	//  Add the Draw Check box to the 'Object Properties' Panel
	glui_window->add_checkbox_to_panel(op_panel, "Draw", &draw);

	//  Add the Wireframe Check box to the 'Object Properties' Panel
	glui_window->add_checkbox_to_panel(op_panel, "Wireframe", &wireframe);

	//  Add a separator
	glui_window->add_separator_to_panel(op_panel);

	//  Add the Color listbox to the 'Object Properties' Panel
	GLUI_Listbox *color_listbox = glui_window->add_listbox_to_panel(op_panel,
		"Color", &listbox_item_id, COLOR_LISTBOX, glui_callback);

	//  Add the items to the listbox
	color_listbox->add_item(1, "Black");
	color_listbox->add_item(2, "Blue");
	color_listbox->add_item(3, "Cyan");
	color_listbox->add_item(4, "Dark Grey");
	color_listbox->add_item(5, "Grey");
	color_listbox->add_item(6, "Green");
	color_listbox->add_item(7, "Light Grey");
	color_listbox->add_item(8, "Magenta");
	color_listbox->add_item(9, "Orange");
	color_listbox->add_item(10, "Pink");
	color_listbox->add_item(11, "Red");
	color_listbox->add_item(12, "White");
	color_listbox->add_item(13, "Yellow");

	//  Select the White Color by default
	color_listbox->set_int_val(12);

	//---------------------------------------------------------------------
	// 'Object Type' Panel
	//---------------------------------------------------------------------

	//  Add the 'Object Type' Panel to the GLUI window
	GLUI_Rollout *ot_rollout = glui_window->add_rollout("Object Type");

	//  Create radio button group
	GLUI_RadioGroup *ot_group = glui_window->add_radiogroup_to_panel
	(ot_rollout, &radiogroup_item_id, OBJECTYPE_RADIOGROUP, glui_callback);

	//  Add the radio buttons to the radio group
	glui_window->add_radiobutton_to_group(ot_group, "Cube");
	glui_window->add_radiobutton_to_group(ot_group, "Sphere");
	glui_window->add_radiobutton_to_group(ot_group, "Cone");
	glui_window->add_radiobutton_to_group(ot_group, "Torus");
	glui_window->add_radiobutton_to_group(ot_group, "Dodecahedron");
	glui_window->add_radiobutton_to_group(ot_group, "Octahedron");
	glui_window->add_radiobutton_to_group(ot_group, "Tetrahedron");
	glui_window->add_radiobutton_to_group(ot_group, "Icosahedron");
	glui_window->add_radiobutton_to_group(ot_group, "Teapot");

	//---------------------------------------------------------------------
	// 'Transformation' Panel
	//---------------------------------------------------------------------

	//  Add the 'Transformation' Panel to the GLUI window
	GLUI_Panel *transformation_panel = glui_window->add_panel("Transformation");

	//  Create transformation panel 1 that will contain the Translation controls
	GLUI_Panel *transformation_panel1 = glui_window->add_panel_to_panel(transformation_panel, "");

	//  Add the xy translation control
	GLUI_Translation *translation_xy = glui_window->add_translation_to_panel(transformation_panel1, "Translation XY", GLUI_TRANSLATION_XY, translate_xy, TRANSLATION_XY, glui_callback);

	//  Set the translation speed
	translation_xy->set_speed(0.005);

	//  Add column, but don't draw it
	glui_window->add_column_to_panel(transformation_panel1, false);

	//  Add the z translation control
	GLUI_Translation *translation_z = glui_window->add_translation_to_panel(transformation_panel1, "Translation Z", GLUI_TRANSLATION_Z, &translate_z, TRANSLATION_Z, glui_callback);

	//  Set the translation speed
	translation_z->set_speed(0.005);

	//  Create transformation panel 2 that will contain the rotation and spinner controls
	GLUI_Panel *transformation_panel2 = glui_window->add_panel_to_panel(transformation_panel, "");

	//  Add the rotation control
	glui_window->add_rotation_to_panel(transformation_panel2, "Rotation", rotation_matrix, ROTATION, glui_callback);

	//  Add separator
	glui_window->add_separator_to_panel(transformation_panel2);

	//  Add the scale spinner
	GLUI_Spinner *spinner = glui_window->add_spinner_to_panel(transformation_panel2, "Scale", GLUI_SPINNER_FLOAT, &scale, SCALE_SPINNER, glui_callback);

	//  Set the limits for the spinner
	spinner->set_float_limits(-4.0, 4.0);

	//---------------------------------------------------------------------
	// 'Quit' Button
	//---------------------------------------------------------------------

	//  Add the Quit Button
	glui_window->add_button("Quit", QUIT_BUTTON, glui_callback);

	//  Let the GLUI window know where its main graphics window is
	glui_window->set_main_gfx_window(main_window);
}

//-------------------------------------------------------------------------
//  GLUI callback function.
//-------------------------------------------------------------------------
void glui_callback(int control_id)
{
	//  Notify that this is a GLUI Callback
	printf("GLUI: ");

	//  Behave based on control ID
	switch (control_id)
	{
		//  Color Listbox item changed
	case COLOR_LISTBOX:

		printf("Color List box item changed: ");

		switch (listbox_item_id)
		{
			//  Select black color
		case 1:
			color[0] = 0 / 255.0;
			color[1] = 0 / 255.0;
			color[2] = 0 / 255.0;
			break;
			//  Select blue color
		case 2:
			color[0] = 0 / 255.0;
			color[1] = 0 / 255.0;
			color[2] = 255 / 255.0;
			break;
			//  Select cyan color
		case 3:
			color[0] = 0 / 255.0;
			color[1] = 255 / 255.0;
			color[2] = 255 / 255.0;
			break;
			//  Select dark grey color
		case 4:
			color[0] = 64 / 255.0;
			color[1] = 64 / 255.0;
			color[2] = 64 / 255.0;
			break;
			//  Select grey color
		case 5:
			color[0] = 128 / 255.0;
			color[1] = 128 / 255.0;
			color[2] = 128 / 255.0;
			break;
			//  Select green color
		case 6:
			color[0] = 0 / 255.0;
			color[1] = 255 / 255.0;
			color[2] = 0 / 255.0;
			break;
			//  Select light gray color
		case 7:
			color[0] = 192 / 255.0;
			color[1] = 192 / 255.0;
			color[2] = 192 / 255.0;
			break;
			//  Select magenta color
		case 8:
			color[0] = 192 / 255.0;
			color[1] = 64 / 255.0;
			color[2] = 192 / 255.0;
			break;
			//  Select orange color
		case 9:
			color[0] = 255 / 255.0;
			color[1] = 192 / 255.0;
			color[2] = 64 / 255.0;
			break;
			//  Select pink color
		case 10:
			color[0] = 255 / 255.0;
			color[1] = 0 / 255.0;
			color[2] = 255 / 255.0;
			break;
			//  Select red color
		case 11:
			color[0] = 255 / 255.0;
			color[1] = 0 / 255.0;
			color[2] = 0 / 255.0;
			break;
			//  Select white color
		case 12:
			color[0] = 255 / 255.0;
			color[1] = 255 / 255.0;
			color[2] = 255 / 255.0;
			break;
			//  Select yellow color
		case 13:
			color[0] = 255 / 255.0;
			color[1] = 255 / 255.0;
			color[2] = 0 / 255.0;
			break;
		}

		printf("Item %d selected.\n", listbox_item_id);

		break;

		//  A Radio Button in the radio group is selected
	case OBJECTYPE_RADIOGROUP:

		printf("Radio Button %d selected.\n", radiogroup_item_id);

		break;

		//  Translation XY control
	case TRANSLATION_XY:

		printf("Translating X and Y coordinates: ");
		printf("X: %f, Y: %f.\n", translate_xy[0], translate_xy[1]);

		break;

		//  Translation Z control
	case TRANSLATION_Z:

		printf("Translating Z coordinate: ");
		printf("Z: %f.\n", translate_z);

		break;


		//  Scaling
	case SCALE_SPINNER:

		printf("Scaling Object: %f.\n", scale);

		break;

		//  Quit Button clicked
	case QUIT_BUTTON:

		printf("Quit Button clicked... Exit!\n");

		exit(1);

		break;

	}
}

//-------------------------------------------------------------------------
//  Idle Callback function.
//
//  Set the main_window as the current window to avoid sending the
//  redisplay to the GLUI window rather than the GLUT window. 
//  Call the Sleep function to stop the GLUI program from causing
//  starvation.
//-------------------------------------------------------------------------
void idle()
{
	glutSetWindow(main_window);
	glutPostRedisplay();
	Sleep(50);
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
	printf("V/vy:   increase decrease fluid viscosity\n");
	printf("x:     toggle drawing matter on/off\n");
	printf("y:     toggle drawing hedgehogs on/off\n");
	printf("m:     toggle thru scalar coloring\n");
	printf("a:     toggle the animation on/off\n");
	printf("q:     quit\n\n");

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(500,500);
	main_window = glutCreateWindow(window_title);
	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutIdleFunc(do_one_simulation_step);
	glutKeyboardFunc(keyboard);
	glutMotionFunc(drag);
	
	init_simulation(DIM);	//initialize the simulation data structures	

//	glutInitWindowSize(300, 50);
//	glutCreateWindow("value");
//	glutDisplayFunc(display2);

//  Setup all GLUI stuff
//	setupGLUI();

	glutMainLoop();			//calls do_one_simulation_step, keyboard, display, drag, reshape
	return 0;
}
