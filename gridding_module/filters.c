#ifndef _FFT_C
#define _FFT_C  
#include "fft.h"
#endif

#define PI 3.141592653589793   
#define myAbs(X) ((X)<0 ? -(X) : (X))



/*******************************************************************/
/*******************************************************************/
/*
 * PARZEN FILTER
 */
float parzenFilter( float x ){
    
  float fm = 0.5;
  
  float ffm;
  
  ffm = myAbs( x )/fm;
  
  if ( myAbs( x ) <= fm/2.)  
    return (float)( 1 - 6*ffm*ffm*( 1 - ffm ));
  
  else if ( myAbs( x ) > fm/2. && myAbs( x ) <= fm ) 
    return (float)( 2 * (1-ffm) * (1-ffm) * (1-ffm) );
  
  else 
    return 0.0f;
}



/*
 *  SHEPP-LOGAN FILTER
 */
float sheppLoganFilter( float x ){	/* Shepp-Logan filter */

  float fm = 0.5; 
  
  float ffm;
  
  ffm = x/(2.*fm);
  
  if ( x==0 )
    return 1.0;
  
  else if ( x<fm )
    return (float)( sin( PI * ffm )/( PI * ffm ) );  /* Shepp-Logan window */
  
  else
    return 0.0;
} 



/*
 * RAMP FILTER
 */
void rampFilter( int size , float *rampArray ) {
	
  int i, j, c;
  int sizeH = (int)( size * 0.5 );
  
  for ( i=0,j=0 ; i<size ; i++,j+=2 ) {
    c = i - ( sizeH-1 );
    
    /* Real component */
    if ( c == 0 ) 
      rampArray[j] = 0.25;
    
    else if ( c % 2 == 0)
      rampArray[j] = 0.0;
    
    else
      rampArray[j] = -1 / ( PI*PI*c*c );
    
    /* Imaginary component */
    rampArray[j+1] = 0;
  }
}



/*
 * FILTER FUNCTION
 */
void calcFilter( float *filter, unsigned long nang, unsigned long N , float center , int flag_filter )
{ 
  long i;
  long j;   // Index running on real part of each component of the filter array
  long k;   // Index running on spatial frequencies
  unsigned long N2 = 2 * N;    
  float x; // Complex exponential whose argument depends on the center
	    //    of rotation and the selected spatial frequency
  float filterWeight; // Select a number going from 0 to 1 as weight for the filter
		       // the default value is 1.0
  float rtmp1 = (float)( 2*PI*center/N ); // Phase term related to the center of rotation
				            // to be multiplied with each spatial frequency
  float rtmp2;
  float tmp;
  float norm = (float)( PI/N/nang );
  float *rampArray;


  filterWeight = 1.0;

  if( flag_filter ){
    //printf("\nSelected filter: SHEPP-LOGAN + RAMP\n");

    rampArray = (float *)calloc( N2 , sizeof(float) );
    
    if ( flag_filter == 1 ) {    
      for( i=0 ; i<N2 ; i++ )
	    rampArray[i] = 0.0;
      
      rampFilter( N , rampArray );

      //four1( rampArray-1 , N , 1 );
      myFour1( rampArray , N , 1 );

      for ( i=0,j=0 ; i<N ; i++,j+=2 )
	    rampArray[i] = sqrt( ( rampArray[j] * rampArray[j] ) +	\
		                     ( rampArray[j+1] * rampArray[j+1] ) );

      while( i<N2 ){
	    rampArray[i] = 0.0;
	    i++;
      }
    }
  }
  //else
  //  printf("\nNo filtering selected\n");

  
  for( j=0,k=0 ; j<N ; j+=2,k++ ){
    x = k * rtmp1;
    
    // Apply filter ( shepp-logan or parzen ) + ramp filter
    if ( flag_filter == 1 )
      rtmp2 = ( 1 - filterWeight + filterWeight * sheppLoganFilter( (float)k/N ) * rampArray[k] ) * norm;

    // Apply no filter
    else
      rtmp2 = 1.0;

    // Filter value + phase term due to the center of rotation
    filter[j] = rtmp2 * cos(x);
    filter[j+1] = -rtmp2 * sin(x);
  }

  if ( flag_filter == 1 )
    free( rampArray );
}
