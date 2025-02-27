#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "nmath.h"
#define pi M_PI
#ifdef NAN
#ifdef INFINITY

SEXP rxkcd_c(SEXP n, SEXP sd) {
    int n_c = INTEGER(n)[0];
    double sd_c = REAL(sd)[0];
    SEXP result;
    PROTECT(result = allocVector(REALSXP, n_c));
    for (int i = 0; i < n_c; i++)
        REAL(result)[i] = runif(0,dnorm(rnorm(0,sd_c),0,sd_c,0));
    UNPROTECT(1);
    return result;
}

double dxkcd1(double x, double sd, int log_p, int swap_end_points){
  double result;
    if (isnan(x)||isnan(sd)||isinf(sd)||sd<=0){
      result = NAN;
      return result;
    }
  if(x==0 && log_p==0 && swap_end_points == 0){
    result =  INFINITY;
    return result;
    }
  if(x==0 && log_p==1 && swap_end_points == 0){
    result =  INFINITY;
    return result;
  }
  if(x==0 && log_p==0 && swap_end_points == 1){
    result =  0;
    return result;
  }
  if(x==0 && log_p==1 && swap_end_points == 1){
    result =  -INFINITY;
    return result;
  }
  if(x==1/(sqrt(2.0*pi)*sd) && log_p==0 && swap_end_points == 0){
    result=0;
    return result;
    }
  if(x==1/(sqrt(2.0*pi)*sd) && log_p==1 && swap_end_points == 0){
    result=-INFINITY;
    return result;
    }
  if(x==1/(sqrt(2.0*pi)*sd) && log_p==0 && swap_end_points == 1){
    result=INFINITY;
    return result;
    }
  if(x==1/(sqrt(2.0*pi)*sd) && log_p==1 && swap_end_points == 1){
    result=INFINITY;
    return result;
    }
  if (x<0 || x>1/(sqrt(2.0*pi)*sd)) {
    result=0;
    return result;
  }
  double const1=-log(sqrt(2.0*pi)*sd*x);
  double const2=log(2.0*sd*sd);
  if (log_p==0 && swap_end_points == 0){
    result= 2.0*sqrt(2.0)*sd*sqrt(-log(sqrt(2.0*pi)*x*sd));
    return result;
  }
  else if(log_p==1 && swap_end_points == 0){
    result= log(2)+0.5*const2+0.5*log(const1);
    return result;
  }
  else if(log_p==0 && swap_end_points == 1){
    result= 2*sqrt(2)*sd*sqrt(-log1p(-sqrt(2*pi)*sd*x));
    return result;
  }
  else {
    result = log(2) + 0.5*const2+0.5*log(-log1p(-sqrt(2*pi)*sd*x));
    return result;
  }
}


SEXP dxkcd_c(SEXP n, SEXP x, SEXP sd, SEXP log_p, SEXP swap_end_points) {
  int n_c = INTEGER(n)[0];
  double *x_c = REAL(x);
  double *sd_c = REAL(sd);
  int log_p_c = INTEGER(log_p)[0];
  int swap_end_points_c = INTEGER(swap_end_points)[0];
  
  SEXP results;
  PROTECT(results = allocVector(REALSXP, n_c));
  
  for (int i = 0; i < n_c; i++){
    REAL(results)[i] = dxkcd1(x_c[i], sd_c[i], log_p_c, swap_end_points_c);
  }
  UNPROTECT(1);
    return results;
}

double pxkcd1(double q, double sd, int log_p, int swap_end_points){
  double result=0;
  if (isnan(q)||isnan(sd)||isinf(sd)||sd<=0){
      result = NAN;
      return result;
  }
  if (q>=1/(sqrt(2*pi)*sd) && log_p==0){
        result =  1;
        return result;
    }
  if (q<=0 && log_p ==0){
        result =  0;
        return result;
    }
  if (q>=1/(sqrt(2*pi)*sd) && log_p ==1){
      result =  0;
      return result;
    }
  if (q<=0 && log_p ==1){
        result = -INFINITY;
        return result;
}

    double hy=dxkcd1(q,sd,0,0)/2;
    double hw=dxkcd1(q,sd,0,1)/2;
    double const1 = 2*sqrt(2*sd*sd*sd*sqrt(2*pi))*2/3.0;
  if (log_p==1 && swap_end_points == 0){
    result=log(2)+pnorm(-hy,0,sd,1,1)+log1p(q*hy/pnorm(-hy,0,sd,1,0));
    return result;
  }
  if(log_p==0 && swap_end_points == 0){
    result=2*pnorm(-hy,0,sd,1,0)+2.0*q*hy;
    return result;
  }
  if(log_p==1 && swap_end_points == 1 && q*sd<= pow(10,-10)){
      result=(log(const1)+1.5*log(q));
    return result;
  }
  if(log_p==1 && swap_end_points == 1  && q*sd>pow(10,-10)){
      result = log(1-2*pnorm(-hw,0,sd,1,0)-2*(1/(sqrt(2*pi)*sd)-q)*hw);
      return result;
  }
  if(log_p==0 && swap_end_points == 1 &&  q*sd<= pow(10,-10)){
      result=const1*pow(q,1.5);
      return result;
  }
  if(log_p==0 && swap_end_points == 1 && q*sd> pow(10,-10)){
      result= 1-(2*pnorm(-hw,0,sd,1,0)+2*(1/(sqrt(2*pi)*sd)-q)*hw);
      return result;
}
    return result;
}

SEXP pxkcd_c(SEXP n, SEXP q, SEXP sd, SEXP log_p, SEXP swap_end_points){
  int n_c = INTEGER(n)[0];
  double *x_c = REAL(q);
  double *sd_c = REAL(sd);
  int log_p_c = INTEGER(log_p)[0];
  int swap_end_points_c = INTEGER(swap_end_points)[0];
  
  SEXP results;
  PROTECT(results = allocVector(REALSXP, n_c));
  
  for (int i = 0; i < n_c; i++){
    REAL(results)[i] = pxkcd1(x_c[i], sd_c[i], log_p_c, swap_end_points_c);
  }
  UNPROTECT(1);
  return results;
}

double func(double q, double p, double sd, int log_p, int swap_end_points){
    return (pxkcd1(q, sd, log_p, swap_end_points)-p);
}

double qxkcd1(double p, double sd, int log_p, int swap_end_points){
  double result;
  if (isnan(p)||isnan(sd)||isinf(sd)||sd<=0){
      result = NAN;
      return result;
  }
  if ((p<0 || p>1)&& log_p==0){
        result =  NAN;
        return result;
    }
  if (p>0 && log_p ==1){
        result =  NAN;
        return result;
    }
  if (isinf(p) && log_p ==1){
      result =  0;
      return result;
    }

   double a = 0;
   double b = 1/(sqrt(2*pi)*sd);
   double e = 0.00000001;
   double c;
   if(func(a,p,sd,log_p,swap_end_points) *
      func(b,p,sd,log_p,swap_end_points) >= 0)
        {
            error("f(lower) * f(upper) >=0");
        }
   c = a;
   
    while ((b-a) >= e)
        {
            c = (a+b)/2;
            if (func(c,p,sd,log_p,swap_end_points) == 0.0){
                break;
            }
            else if (func(c,p,sd,log_p,swap_end_points)*
                     func(a,p,sd,log_p,swap_end_points) < 0){
                    b = c;
            }
            else{
                    a = c;
            }
        }
    return c;
}

SEXP qxkcd_c(SEXP n, SEXP p, SEXP sd, SEXP log_p, SEXP swap_end_points){
  int n_c = INTEGER(n)[0];
  double *x_c = REAL(p);
  double *sd_c = REAL(sd);
  int log_p_c = INTEGER(log_p)[0];
  int swap_end_points_c = INTEGER(swap_end_points)[0];
  
  SEXP results;
  PROTECT(results = allocVector(REALSXP, n_c));
  
  for (int i = 0; i < n_c; i++){
    REAL(results)[i] = qxkcd1(x_c[i], sd_c[i], log_p_c, swap_end_points_c);
  }
  UNPROTECT(1);
  return results;
}

#endif
#endif
