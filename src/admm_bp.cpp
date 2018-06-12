#include "RcppArmadillo.h"
#include <sys/time.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

arma::colvec bp_shrinkage(arma::colvec a, const double kappa){
  const int n = a.n_elem;
  arma::colvec y(n,fill::zeros);
  for (int i=0;i<n;i++){
    // first term : max(0, a-kappa)
    if (a(i)-kappa > 0){
      y(i) = a(i)-kappa;
    }
    // second term : -max(0, -a-kappa)
    if (-a(i)-kappa > 0){
      y(i) = y(i) + a(i) + kappa;
    }
  }
  return(y);
}

unsigned long long get_time_ms(){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  unsigned long long millisecondsSinceEpoch =
    (unsigned long long)(tv.tv_sec) * 1000 +
    (unsigned long long)(tv.tv_usec) / 1000;
  return(millisecondsSinceEpoch);
}


/*
 * Basis Pursuit via ADMM (from Stanford)
 * URL : https://web.stanford.edu/~boyd/papers/admm/basis_pursuit/basis_pursuit.html
 * http://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
 * page 19 : section 3.3.1 : stopping criteria part (3.12).
 */
//' @keywords internal
//' @noRd
// [[Rcpp::export]]
Rcpp::List admm_bp(const arma::mat& A, const arma::colvec& b, arma::colvec& xinit,
                   const double reltol, const double abstol, const int maxiter,
                   const double rho, const double alpha){
  unsigned long long start_time = get_time_ms();
  // 1. get parameters
  const int m = A.n_rows;
  const int n = A.n_cols;

  // 2. set ready
  arma::colvec x(n,fill::zeros);
  arma::colvec z(n,fill::zeros);
  arma::colvec u(n,fill::zeros);
  arma::colvec zold(n,fill::zeros);
  arma::colvec x_hat(n,fill::zeros);

  // 3. precompute static variables for x-update
  std::cout << std::fixed << get_time_ms() / 1000.0;
  std::cout << " Precomputing static vars...\n";
  arma::vec n1s(n,fill::ones);
  arma::mat AAt  = A*A.t();
  arma::mat P    = diagmat(n1s) - A.t()*solve(AAt,A);
  arma::colvec q = A.t()*solve(AAt,b);

  // 4. iteration
  arma::vec h_objval(maxiter,fill::zeros);
  arma::vec h_r_norm(maxiter,fill::zeros);
  arma::vec h_s_norm(maxiter,fill::zeros);
  arma::vec h_eps_pri(maxiter,fill::zeros);
  arma::vec h_eps_dual(maxiter,fill::zeros);
  arma::vec h_ts(maxiter, fill::zeros);

  double sqrtn = sqrt(static_cast<double>(n));
  int k;
  std::cout << std::fixed << get_time_ms() / 1000.0;
  std::cout << " Starting ADMM iterations...\n";
  for (k=0;k<maxiter;k++){
    h_ts(k) = (double) (get_time_ms() - start_time) / 1000.0;

    // 4-1. update 'x'
    x = P*(z-u) + q;

    // 4-2. update 'z' with relaxation
    zold = z;
    x_hat = alpha*x + (1 - alpha)*zold;
    z = bp_shrinkage(x_hat + u, 1/rho);
    u = u + (x_hat - z);

    // 4-3. dianostics, reporting
    h_objval(k) = norm(x,1);
    h_r_norm(k) = norm(x-z);
    h_s_norm(k) = norm(-rho*(z-zold));
    if (norm(x)>norm(-z)){
      h_eps_pri(k) = sqrtn*abstol + reltol*norm(x);
    } else {
      h_eps_pri(k) = sqrtn*abstol + reltol*norm(-z);
    }
    h_eps_dual(k) = sqrtn*abstol + reltol*norm(rho*u);

    // 4-4. termination
    if ((h_r_norm(k) < h_eps_pri(k))&&(h_s_norm(k)<h_eps_dual(k))){
      break;
    }
  }

  // 5. report results
  List output;
  output["x"] = x;             // coefficient function
  output["objval"] = h_objval; // |x|_1
  output["k"] = k;             // number of iterations
  output["r_norm"] = h_r_norm;
  output["s_norm"] = h_s_norm;
  output["eps_pri"] = h_eps_pri;
  output["eps_dual"] = h_eps_dual;
  output["ts"] = h_ts;
  return(output);
}
