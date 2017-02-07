/********************************************************************  
 ********************************************************************
 ***                                                              ***
 ***         REGRIDDING BACKPROJECTION OPERATOR FOR PYNNGRID      ***
 ***                                                              *** 
 ***             Written by F. Arcadu on the 18/03/2014           ***
 ***                                                              ***
 ********************************************************************
 ********************************************************************/




/********************************************************************
 ********************************************************************
 ***                                                              ***
 ***                             HEADERS                          ***
 ***                                                              ***
 ********************************************************************
 ********************************************************************/

#include <math.h>

#ifndef _FFT_C
#define _FFT_C  
#include "fft.h"
#endif

#ifndef _PWSF_LIB
#define _PSWF_LIB
#include "pswf.h"
#endif

#ifndef _FILTERS_LIB
#define _FILTERS_LIB
#include "filters.h"
#endif

#ifndef _FFTW3_LIB
#define _FFTW3_LIB
#include <fftw3.h>
#endif


/********************************************************************
 ********************************************************************
 ***                                                              ***
 ***                             MACROS                           ***
 ***                                                              ***
 ********************************************************************
 ********************************************************************/

#define Cnvlvnt(X) ( wtbl[ (int)(X+0.5) ])
#define myAbs(X) ((X)<0 ? -(X) : (X))
#define PI 3.141592653589793
#define C 7.0 




/********************************************************************
 ********************************************************************
 ***                                                              ***
 ***                         BACKPROJECTION                       ***
 ***                                                              ***
 ********************************************************************
 ********************************************************************/

void gridrec_backproj( float *sino , int npix , int nang , float *angles , 
                       float *param , float *filter , float *rec, char *fftwfn ) {
    
  /*
   *   Define variables
   */

  int pdim;                 //  Number of zero-padded pixels in order to have a number of pixels power of 2
  int padleft;              //  Pixel position of the last padding zero on the left of the actual projection
  int padright;             //  Pixel position of the first padding zero on the right of the actual projection
  int pdim_d;               //  Double number of "pdim" pixels
  int pdim_h;               //  Half number of "pdim" pixels
  int padfactor;            //  Additional edge-padding factor
  int flag_filter;          //  Flag to specify whether to use the external filter or the standard built-in one

  unsigned long i, j, k, w, n, index;
  unsigned long iul, ivl, iuh, ivh, iv, iu, ivh_d, ivl_d;
  unsigned long ltbl;       // Number of PSWF elements
  
  long ustart, vstart, ufin, vfin;
  
  float Cdata1R, Cdata1I, Cdata2R, Cdata2I;
  float Ctmp1R, Ctmp1I, Ctmp2R, Ctmp2I, Ctmp3R, Ctmp3I;     
  float U, V;             // iariables referred to the Fourier cartesian grid
  float lconv;            // Size of the convolution window
  float lconv_h;           // Half size of the convolution window
  float rtmp;
  float scaling;          // Convert frequencies to grid units
  float tblspcg;
  float convolv;
  float corrn_u, corrn;
  float antinorm_factor;  // Factor to appropriately normalize the IFFT 2D
  float ctr;              // Variable to store the center of rotation axis 		  
  fftwf_complex *cproj;           
  float *SINE, *COSE;     
  float *wtbl;            
  float *dwtbl;            
  float *winv;            
  float *work;
  fftwf_complex *H;
  float *filter_stand;
  float x;
  float tmp;
  float norm;
  
  FILE *fp = fopen(fftwfn,"r");
  if(fp){
    fftwf_import_wisdom_from_file(fp); // Load wisdom file for faster FFTW
    fclose(fp);
  }

  
  /*
   *   Calculate number of operative pixels
   *   This number is equal to smallest power of 2 which >= to the original
   *   number of pixels multiplied by the edge-padding factor 2
   */

  padfactor = 2;
  pdim = (int) pow( 2 , (int)( ceil( log10( npix )/log10(2) ) )) * padfactor;
  padleft = (int) ( ( pdim - npix ) * 0.5 );
  padright = (int) ( padleft + npix );
  pdim_d = 2 * pdim;
  pdim_h = 0.5 * pdim;



  /*
   *   Allocate memory for the complex array storing first each projection
   *   and, then, its Fast Fourier Transform
   */

  cproj = (fftwf_complex *)fftwf_malloc( pdim*sizeof(fftwf_complex));



  /*
   *   Enable scaling
   *   Here it is disabled
   */

  rtmp, scaling = 1.0;
  


  /* 
   *   Initialize look-up-table for sin and cos of the projection angles
   */

  SINE = (float *)calloc( nang , sizeof(float) );
  COSE = (float *)calloc( nang , sizeof(float) );  
  lutTrig(  nang , angles , SINE , COSE );

  

  /*
   *   Initialize look-up-table for the PSWF interpolation kernel and
   *   for the correction matrix
   */
  
  lconv = (float)( 2 * C * 1.0 / PI ); 
  lconv_h = lconv * 0.5; 
  ltbl = 2048;
  tblspcg = 2 * ltbl / lconv;  
  wtbl = (float *)calloc( ltbl + 1 , sizeof(float) );
  dwtbl = (float *)calloc( ltbl + 1 , sizeof(float) );  
  winv = (float *)calloc( pdim+1 , sizeof(float) );
  work = (float *)calloc( (int)lconv + 1 , sizeof(float) );
  lutPswf( ltbl , pdim_h , wtbl , dwtbl , winv ); 

  
  
  /*
   *   Get center of rotation axis
   */
  ctr = (float) param[0];
  if ( ctr == 0 )
    ctr = npix * 0.5;
  if( pdim != npix )
    ctr += (float) padleft;
   

  flag_filter = param[1];


  /*
   *   Use standard filter
   */
  
  if( flag_filter ){
    filter_stand = (float *)calloc( pdim_d , sizeof(float) );
    calcFilter( filter_stand , nang , pdim , ctr , flag_filter );      
  } 
  


  /*
   *   Multiply input filter array per complex exponential in order
   *   to correct for the center of rotation axis
   */
  
  else{
    tmp = (float)( 2 * PI * ctr / pdim );
    norm = (float)( PI / pdim / nang );

    for( j=0,k=0 ; j<pdim ; j+=2,k++ ){
        x = k * tmp;
        float fValue = filter[j];
        filter[j] = fValue*norm * cos(x);
        filter[j+1] = -fValue*norm * sin(x);
    }
  }
  

  
  /*
   *   Allocate memory for cartesian Fourier grid
   */
  
  H = (fftwf_complex*)fftwf_malloc(pdim*pdim*sizeof(fftwf_complex));
  for(n=0;n<pdim*pdim;n++){
      H[n][0]=0;
      H[n][1]=0;
  }
  
  fftwf_plan p1 = fftwf_plan_dft_1d(pdim, cproj, cproj, FFTW_FORWARD, FFTW_ESTIMATE);

  /*
   *   Interpolation of the polar Fourier grid with PSWF 
   */
  
  for( n=0 ; n<nang ; n++ ){ 
   
    /*
     *   Store each projection inside "cproj" and, at the same
     *   time, perform the edge-padding of the projection
     */
    
    i = 0;
    j = 0;
    k = 0;

    while( i < padleft ){
      cproj[j][0] = sino[ n * npix ];
      cproj[j][1] = 0.0;
      i += 1;
      j += 1;
    }

    while( i < padright ){
      cproj[j][0] = sino[ n*npix + k ];
      cproj[j][1] = 0.0;
      i += 1;
      j += 1;
      k += 1;
    }

    while( i < pdim ){
      cproj[j][0] = sino[ n * npix + npix - 1 ];
      cproj[j][1] = 0.0;
      i += 1;
      j += 1;
    }

    /*
     *   Perform 1D FFT of the projection
     */
     
     fftwf_execute(p1);
    
    /*
     *   Loop on the first half Fourier components of
     *   each FFT-transformed projection (one exploits
     *   here the hermitianity of the FFT-array related 
     *   to an original real array)
     */
    
    for( j=0, w=0 ; j < pdim_h ; j++, w++ ) {  
      
      if( flag_filter ){
	    Ctmp1R = filter_stand[2*j];
	    Ctmp1I = filter_stand[2*j+1];
      }
      else{
	    Ctmp1R = filter[2*j];
	    Ctmp1I = filter[2*j+1];       
      }
	
	  Ctmp2R = cproj[j][0];
	  Ctmp2I = cproj[j][1];
	
	  if( j!=0 ){
	    Ctmp3R = cproj[pdim-j][0];
	    Ctmp3I = cproj[pdim-j][1];
	
        Cdata1R = Ctmp1R * Ctmp3R  -  Ctmp1I * Ctmp3I;
        Cdata1I = Ctmp1R * Ctmp3I  +  Ctmp1I * Ctmp3R;

        Ctmp1I = -Ctmp1I;

        Cdata2R = Ctmp1R * Ctmp2R  -  Ctmp1I * Ctmp2I;
        Cdata2I = Ctmp1R * Ctmp2I  +  Ctmp1I * Ctmp2R;
	  }

	  else {
	    Cdata1R = Ctmp1R * Ctmp2R;
	    Cdata1I = Ctmp1R * Ctmp2I;  
	    Cdata2R = 0.0;
	    Cdata2I = 0.0;
	  }
      

      /*
       *   Get polar coordinates
       */
     
      U = ( rtmp = scaling * w ) * COSE[n] + pdim_h; 
      V = rtmp * SINE[n] + pdim_h;
     

      /*
       *   Get interval of the cartesian coordinates
       *   of the points receiving the contribution from
       *   the selected polar Fourier sample
       */
     
      iul = (long)ceil( U - lconv_h ); iuh = (long)floor( U + lconv_h );
      ivl = (long)ceil( V - lconv_h ); ivh = (long)floor( V + lconv_h );

      if ( iul<0 ) iul = 0; if ( iuh >= pdim ) iuh = pdim-1;   
      if ( ivl<0 ) ivl = 0; if ( ivh >= pdim ) ivh = pdim-1;
      
      ivl_d = 2 * ivl;
      ivh_d = 2 * ivh;
     

      /*
       *   Get PSWF convolvent values
       */
      for (iv = ivl, k=0; iv <= ivh; iv++, k++)
	    work[k] = Cnvlvnt( myAbs( V - iv ) * tblspcg );
     

     /*
      *   Calculate the contribution of each polar Fourier point
      *   for all the neighbouring cartesian Fourier points 
      */
    
      for( iu=iul ; iu<=iuh ; iu++ ){
	    rtmp = Cnvlvnt( myAbs( U - iu ) * tblspcg );
	    
        for( iv=ivl , k=0 ; iv<= ivh ; iv++,k++ ){
	      convolv = rtmp * work[k];
          
          if (iu!=0 && iv!=0 && w!=0) { 
		    H[iu * pdim + iv][0] += convolv * Cdata1R;
            H[iu * pdim + iv][1] += convolv * Cdata1I;
            H[(pdim-iu) * pdim + pdim - iv][0] += convolv * Cdata2R;
            H[(pdim-iu) * pdim + pdim - iv][1] += convolv * Cdata2I;
          } 
          else {
            H[iu * pdim + iv][0] += convolv * Cdata1R;
            H[iu * pdim + iv][1] += convolv * Cdata1I;
          }
        } // End loop on y-coordinates of the cartesian neighbours 
	  } // End loop on x-coordinates of the cartesian neighbours
    } // End loop on transform data   
  } // End for loop on angles
  


  /*
   *   Perform 2D FFT of the cartesian Fourier Grid
   */
    fftwf_plan p =  fftwf_plan_dft_2d(pdim, pdim,
                                H, H,
                                FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    
    fftwf_destroy_plan(p1);

  /*
   *  Multiplication for the PSWF correction matrix
   */
  
  antinorm_factor = (float)( 0.5 ); 

  j = 0;
  ustart = pdim - pdim_h;
  ufin = pdim;
 
  while (j < pdim){    
    for( iu = ustart ; iu < ufin ; j++ , iu++ ){
      corrn_u = winv[j];
      k = 0;
      vstart = pdim - pdim_h ;
      vfin = pdim;
      
      while( k < pdim ){
	    for( iv = vstart ; iv < vfin ; k++ , iv++ ) {
	      corrn = corrn_u * winv[k];
    
	      /*
           *   Select the centered square npix * npix
           */
     
     	  if( padleft <= j && j < padright && padleft <= k && k < padright ){
	        index = ( npix - 1 - (k-padleft) ) * npix + (j-padleft);
	        rec[index] = corrn * H[iu * pdim + iv][0] * antinorm_factor;
	      }	  
	    }
      
	    if (k < pdim) {
	      vstart = 0; 
      	  vfin = pdim_h;
      	}
      } // End while loop on k
    } // End for loop on iu

    if (j < pdim) {
      ustart = 0;
      ufin = pdim_h;
    }
  } // End while loop on j
  


  /*
   *  Free memory
   */

  
  free( SINE );
  free( COSE );
  free( wtbl );
  free( dwtbl );
  free( winv );
  free( work );
  fftwf_free( cproj );
  fftwf_free( H );
  if( flag_filter )
      free( filter_stand );
  

  
  return;
}  

void create_fftw_wisdom_file(char *fn, int npix){
    int padfactor = 2;
    int pdim = (int) pow( 2 , (int)( ceil( log10( npix )/log10(2) ) )) * padfactor;
    fftwf_complex *H = (fftwf_complex*)fftwf_malloc(pdim*pdim*sizeof(fftwf_complex));
    fftwf_complex *H2 = (fftwf_complex*)fftwf_malloc(pdim*sizeof(fftwf_complex));
    FILE *fp = fopen(fn,"r");
    if(fp){
        fftwf_import_wisdom_from_file(fp); // Load wisdom file for faster FFTW
        fclose(fp);
    }
    fftwf_plan p =  fftwf_plan_dft_2d(pdim, pdim,
                                H, H,
                                FFTW_BACKWARD, FFTW_MEASURE);
    fftwf_plan p1 =  fftwf_plan_dft_1d(pdim,
                                H2,H2,
                                FFTW_FORWARD, FFTW_MEASURE);
                                
    fp = fopen(fn,"w");
    if(fp){
        fftwf_export_wisdom_to_file(fp);
        fclose(fp);
    }
    fftwf_destroy_plan(p);
    fftwf_destroy_plan(p1);
    fftwf_free(H);
    fftwf_free(H2);
}
