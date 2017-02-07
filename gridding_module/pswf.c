/**********************************************************
 **********************************************************
 ***                                                    ***
 ***                        MACROS                      ***
 ***                                                    ***
 **********************************************************
 **********************************************************/

#define C 7.0
#define NT 20
#define PI 3.141592653589793
#define Cnvlvnt(X) ( wtbl[ (int)(X+0.5) ])




/**********************************************************
 **********************************************************
 ***                                                    ***
 ***    COMPUTE OPERATIONS WITH LEGENDRE POLYNOMIALS    ***
 ***                                                    ***
 **********************************************************
 **********************************************************/

float legendre( int n , float *coefs , float x )

    /***************************************************
     *                                                  *
     *    Compute SUM(coefs(k)*P(2*k,x), for k=0,n/2)   *
     *                                                  *
     *    where P(j,x) is the jth Legendre polynomial   *
     *                                                  *
     ***************************************************/
{
  float penult, last, new_, y;
  int j,k,even;
  
  y=coefs[0];
  penult=1.;
  last=x;
  even=1;
  k=1;
  for( j=2 ; j<=n ; j++ ){
    new_=(x*(2*j-1)*last-(j-1)*penult)/j;
    
    if(even){
      y+=new_*coefs[k];
      even=0;
      k++;
    }
    else
      even=1;
    
    penult=last;
    last=new_;
  }
 
  return y;  
}



/**********************************************************
 **********************************************************
 ***                                                    ***
 *** LOOK-UP TABLE FOR CONVOLVENT AND FINAL CORRECTION  ***
 ***                                                    ***
 **********************************************************
 **********************************************************/

void lutPswf( long ltbl , long linv , float *wtbl , float *dwtbl , float *winv )

    /*************************************************************/
    /*           Set up lookup tables for convolvent             */
    /*************************************************************/
{
  float polyz,norm,fac;
  long i;
  
  float coefs[11] = { 0.5767616E+02,	-0.8931343E+02,	 0.4167596E+02,
		              -0.1053599E+02,	 0.1662374E+01,	-0.1780527E-00,
		               0.1372983E-01,	-0.7963169E-03,	 0.3593372E-04,
		              -0.1295941E-05,	 0.3817796E-07};
  
  float lmbda = 0.99998546;	
  
  polyz = legendre( NT , coefs , 0. );
  
  
  wtbl[0] = 1.0;
  for( i=1 ; i<=ltbl ; i++ ){
      wtbl[i] = legendre( NT , coefs , (float)i / ltbl ) / polyz;
      dwtbl[i] = wtbl[i] - wtbl[i-1];
  }
  
  fac = (float) ( ltbl / (linv+0.5) );
  norm = (float) sqrt( PI / 2 / C / lmbda );  
  
  /* Note the final result at end of Phase 3 contains the factor, 
     norm^2.  This incorporates the normalization of the 2D
     inverse FFT in Phase 2 as well as scale factors involved
     in the inverse Fourier transform of the convolvent.
     7/7/98 			*/

  winv[linv] = norm / Cnvlvnt(0.);

  
    for( i=1 ; i<=linv ; i++ ){
      norm =- norm; 
      /* Minus sign for alternate entries
	     corrects for "natural" data layout
	     in array H at end of Phase 1.  */
      winv[linv+i] = winv[linv-i] =  norm / Cnvlvnt(i*fac);
    }   
}




/**********************************************************
 **********************************************************
 ***                                                    ***
 ***     CREATE LOOK-UP TABLE OF SIN AND COS VALUES     ***
 ***                                                    ***
 **********************************************************
 **********************************************************/

void lutTrig( int nang , float *angles , float *SINE , float *COSE ){
  int j;
  
  float theta;
  float degtorad = PI/(float)180;
  
  for( j=0 ; j<nang ; j++ ){
    theta = degtorad * angles[j];
    SINE[j] = sin(theta);
    COSE[j] = cos(theta);
  }
}
