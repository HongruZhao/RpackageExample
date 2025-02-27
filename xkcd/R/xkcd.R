rxkcd <- function(n,sd=1){
  out <- .Call("rxkcd_c",n=as.integer(n),sd=as.double(sd))
  return (out)
}

dxkcd=function(x, sd=1, log.p=FALSE,swap.end.points = FALSE){
  ## Make x and sd have the same length.
  x_len=length(x)
  sd_len=length(sd)
  if(x_len<sd_len){
    x=rep(x,length.out=sd_len)
  }
  else{
    sd=rep(sd,length.out=x_len)
  }
  out <- .Call("dxkcd_c",n=as.integer(length(x)), x = as.double(x),
               sd=as.double(sd), log_p=as.integer(log.p), 
               swap_end_points=as.integer(swap.end.points))
  return (out)
}


pxkcd=function(q, sd=1, log.p = FALSE, swap.end.points = FALSE){
  q_len=length(q)
  sd_len=length(sd)
  if(q_len<sd_len){
    q=rep(q,length.out=sd_len)
  }
  else{
    sd=rep(sd,length.out=q_len)
  }
  out <- .Call("pxkcd_c",n=as.integer(length(q)), q = as.double(q),
               sd=as.double(sd), log_p=as.integer(log.p), 
               swap_end_points=as.integer(swap.end.points))
  return (out)
}


qxkcd=function(p, sd=1, log.p = FALSE, swap.end.points = FALSE){
  p_len=length(p)
  sd_len=length(sd)
  if(p_len<sd_len){
    p=rep(p,length.out=sd_len)
  }
  else{
    sd=rep(sd,length.out=p_len)
  }
  out <- .Call("qxkcd_c",n=as.integer(length(p)), p = as.double(p),
               sd=as.double(sd), log_p=as.integer(log.p), 
               swap_end_points=as.integer(swap.end.points))
  return (out)
}


