#开局
set.seed(21068121)
options(digits = 4)
library(mvtnorm)
library(MASS)
#生成协方差
generate_covariance_matrix <- function(n, p) {
  cov_matrix <- matrix(0, nrow = p, ncol = p)
  for (i in 1:p) {
    for (j in 1:p) {
      cov_matrix[i, j] <- 0.5 ^ abs(i - j)
    }
  }
  return(cov_matrix)
}
# 生成数据
generate_data <- function(n, p, dist_type, outlier_type, outlier_prop) {
  # 生成协方差矩阵
  cov_matrix <- generate_covariance_matrix(n, p)
  
  if (dist_type == "normal") {
    X <- rmvnorm(n, mean = rep(1, p), sigma = cov_matrix)
    if (outlier_type == "offset") {
      outlier_indices <- sample(1:n, size = round(n * outlier_prop))
      X[outlier_indices, ] <- rmvnorm(round(n * outlier_prop), mean = rep(3, p), sigma = cov_matrix)
    } else if (outlier_type == "wave") {
      outlier_indices <- sample(1:n, size = round(n * outlier_prop))
      X[outlier_indices, ] <- rmvnorm(round(n * outlier_prop), mean = rep(1, p), sigma = cov_matrix * 5) 
    } else if (outlier_type == "mixed") {
      outlier_indices <- sample(1:n, size = round(n * outlier_prop))
      X[outlier_indices, ] <- rmvnorm(round(n * outlier_prop), mean = rep(3, p), sigma = cov_matrix * 5)
    }
  } else if (dist_type == "t10") {
    X <- rmvt(n, sigma = cov_matrix, df = 10, delta = rep(1, p))
    if (outlier_type == "offset") {
      outlier_indices <- sample(1:n, size = round(n * outlier_prop))
      X[outlier_indices, ] <- rmvt(round(n * outlier_prop) ,sigma = cov_matrix, df = 10, delta = rep(3, p))
    } else if (outlier_type == "wave") {
      outlier_indices <- sample(1:n, size = round(n * outlier_prop))
      X[outlier_indices, ] <- rmvt(round(n * outlier_prop) ,sigma = 5 * cov_matrix, df = 10, delta = rep(1, p))
    } else if (outlier_type == "mixed") {
      outlier_indices <- sample(1:n, size = round(n * outlier_prop))
      X[outlier_indices, ] <- rmvt(round(n * outlier_prop) ,sigma = 5 * cov_matrix, df = 10, delta = rep(3, p))
    }
  } else if (dist_type == "t3") {
    X <- rmvt(n, sigma = cov_matrix, df = 3, delta = rep(1, p))
    if (outlier_type == "offset") {
      outlier_indices <- sample(1:n, size = round(n * outlier_prop))
      X[outlier_indices, ] <- rmvt(round(n * outlier_prop) ,sigma = cov_matrix, df = 3, delta = rep(3, p))
    } else if (outlier_type == "wave") {
      outlier_indices <- sample(1:n, size = round(n * outlier_prop))
      X[outlier_indices, ] <- rmvt(round(n * outlier_prop) ,sigma = 5 * cov_matrix, df = 3, delta = rep(1, p))
    } else if (outlier_type == "mixed") {
      outlier_indices <- sample(1:n, size = round(n * outlier_prop))
      X[outlier_indices, ] <- rmvt(round(n * outlier_prop) ,sigma = 5 * cov_matrix, df = 3, delta = rep(3, p))
    }
  }
  return(list(X=X,outlier_indices=outlier_indices))
}
#抽样方法
unif_sampling <- function(X, y, r) {
  n <- nrow(X)
  sampled_indices_unif <- sample(1:n, size = r, replace = FALSE)
  X_sampled_unif <- X[sampled_indices_unif, ]
  y_sampled_unif <- y[sampled_indices_unif]
  return(list(X_sampled = X_sampled_unif, y_sampled = y_sampled_unif,sampled_indices_unif =sampled_indices_unif))
}

# 杠杆抽样（LEV）
lev_sampling <- function(X, y, r) {
  # 计算杠杆得分（Leverage Scores）
  hat_matrix <- X %*% solve(t(X) %*% X) %*% t(X)
  hii <- diag(hat_matrix)
  pi_i_LEV <- hii / sum(hii)
  # 进行抽样（Sampling）
  sampled_indices_lev <- sample(1:nrow(X), size = r, prob = pi_i_LEV, replace = FALSE)
  X_sampled_lev <- X[sampled_indices_lev, ]
  y_sampled_lev <- y[sampled_indices_lev]
  return(list(X_sampled = X_sampled_lev, y_sampled = y_sampled_lev,sampled_indices_lev =sampled_indices_lev))
}

# 纯净子集抽样（SUBLEV）
sublev_sampling <- function(X, y, outlier_prop) {
  # 首先进行异常值检测（这里简单假设异常值为与均值偏差较大的值，实际可能需要更复杂的检测方法）
  mu <- colMeans(X)
  sigma <- apply(X, 2, sd)
  outlier_indices <- which(apply(X, 1, function(x) any((x - mu) / sigma > 3)))  # 假设3倍标准差之外为异常值
  # 去除异常值得到纯净子集
  X_pure <- X[-outlier_indices, ]
  y_pure <- y[-outlier_indices]
  # 在纯净子集上进行抽样（这里采用简单随机抽样，与UNIF类似）
  r <- round(nrow(X) * outlier_prop)
  n_pure <- nrow(X_pure)
  sampled_indices_pure <- sample(1:n_pure, size = r, replace = FALSE)
  X_sampled_pure <- X_pure[sampled_indices_pure, ]
  y_sampled_pure <- y_pure[sampled_indices_pure]
  return(list(X_sampled = X_sampled_pure, y_sampled = y_sampled_pure,sampled_indices_pure = sampled_indices_pure))
}

# k - means抽样（KLEV）
klev_sampling <- function(X, y, r) {
  # 假设聚类数为3（可根据实际情况调整）
  k <- 3
  # 进行k - means聚类
  kmeans_result <- kmeans(X, centers = k)
  # 获取聚类结果中样本数量最少的一类的索引
  min_cluster_index <- which.min(table(kmeans_result$cluster))
  # 从数据总体中去除个数最少的一类所包含的样本，得到X_S2
  X_S2 <- X[kmeans_result$cluster!= min_cluster_index, ]
  y_S2 <- y[kmeans_result$cluster!= min_cluster_index]
  # 计算更新的Hat矩阵H_S2（这里简化计算，实际可能需要更复杂的计算）
  H_S2 <- X_S2 %*% solve(t(X_S2) %*% X_S2) %*% t(X_S2)
  # 根据H_S2的对角线元素得到计算样本的抽样概率pi_i_Kmeans_LEV（简化计算，实际可能需要归一化等操作）
  pi_i_Kmeans_LEV <- diag(H_S2)
  # 进行抽样（这里简单随机抽样，实际可能需要根据概率pi_i_Kmeans_LEV进行更复杂的抽样操作）
  sampled_indices_klev <- sample(1:nrow(X_S2), size = r, replace = FALSE)
  X_sampled_klev <- X_S2[sampled_indices_klev, ]
  y_sampled_klev <- y_S2[sampled_indices_klev]
  return(list(X_sampled = X_sampled_klev, y_sampled = y_sampled_klev,sampled_indices_klev = sampled_indices_klev))
}

##两阶段Leverage重要性抽样
#计算马氏距离
mahalanobis_distance <- function(x, center, cov_matrix) {
  n <- nrow(x)
  distances <- numeric(n)
  for (i in 1:n) {
    distances[i] <- t(x[i, ] - center) %*% solve(cov_matrix) %*% (x[i, ] - center)
  }
  return(distances)
}

# 计算有序距离序列的斜率序列向量
calculate_slope_vector <- function(distances, order_vector) {
  n <- length(distances)
  slope_vector <- numeric(n - 1)
  for (i in 1:(n - 1)) {
    slope_vector[i] <- (distances[i + 1] - distances[i]) / (order_vector[i + 1] - order_vector[i])
  }
  return(slope_vector)
}

# 定义类的直径
class_diameter <- function(slope_vector, start, end) {
  class_slopes <- slope_vector[start:(end - 1)]
  mean_slope <- mean(class_slopes)
  diameter <- sum((class_slopes - mean_slope)^2)
  return(diameter)
}

# 定义分类的损失函数
loss_function <- function(slope_vector, partition) {
  k <- length(partition)
  loss <- 0
  for (i in 1:(k - 1)) {
    start <- partition[i]
    end <- partition[i + 1]
    loss <- loss + class_diameter(slope_vector, start, end)
  }
  return(loss)
}

# 寻找最优分割点
find_optimal_partition <- function(slope_vector) {
  n <- length(slope_vector) + 1
  min_loss <- Inf
  optimal_partition <- NULL
  for (j in 2:(n - 1)) {
    partition <- c(1, j, n)
    current_loss <- loss_function(slope_vector, partition)
    if (current_loss < min_loss) {
      min_loss <- current_loss
      optimal_partition <- partition
    }
  }
  return(optimal_partition)
}

# 两阶段Leverage重要性抽样函数
tlev_sampling <- function(X, y, r) {
  x_mean <- colMeans(X)
  x_cov <- cov(X)
  distances <- mahalanobis_distance(X, x_mean, x_cov)
  order_vector <- 1:nrow(X)
  # 有序距离序列的斜率序列向量
  slope_vector <- calculate_slope_vector(distances, order_vector)
  # 最优分割点
  optimal_partition <- find_optimal_partition(slope_vector)
  threshold_distance <- distances[optimal_partition[2]]
  # 划分稳健子集和异常子集
  robust_subset <- X[distances <= threshold_distance, ]
  outlier_subset <- X[distances > threshold_distance, ]
  # 稳健子集的Leverage得分
  x_robust <- as.matrix(robust_subset)
  hat_matrix_robust <- x_robust %*% solve(t(x_robust) %*% x_robust) %*% t(x_robust)
  leverage_scores_robust <- diag(hat_matrix_robust)
  # 归一化处理得入样概率
  sampling_probs_robust <- leverage_scores_robust / sum(leverage_scores_robust)
  # 抽样
  if (nrow(robust_subset) >= r) {
    sampled_indices <- sample(nrow(robust_subset), r, prob = sampling_probs_robust, replace = FALSE)
  } else {
    warning("稳健子集的样本数量小于所需抽样数量，将返回全部稳健子集样本。")
    sampled_indices <- 1:nrow(robust_subset)
  }
  sampled_data <- robust_subset[sampled_indices, ]
  X_sampled <- as.matrix(sampled_data)
  Y_sampled <- y[sampled_indices] 
  hat_matrix_sampled <- X_sampled %*% solve(t(X_sampled) %*% X_sampled) %*% t(X_sampled)
  D <- 1 / sqrt(r * sampling_probs_robust[sampled_indices])
  W <- diag(D) %*% hat_matrix_sampled %*% diag(D)
  tlev_estimate <- solve(t(X_sampled) %*% W %*% X_sampled) %*% t(X_sampled) %*% W %*% Y_sampled
  return(list(X_sampled=X_sampled, y_sampled=Y_sampled,robust_indices=sampled_indices,sampled_data = sampled_data, tlev_estimate = tlev_estimate))
}


# 预测值均方误差（MSE）
mse <- function(y_true, y_pred) {
  mean((y_true - y_pred)^2)
}

# 决定系数（R2）
r2 <- function(y_true, y_pred) {
  y_mean <- mean(y_true)
  1 - sum((y_true - y_pred)^2) / sum((y_true - y_mean)^2)
}

# 系数估计值平均绝对误差（MAE）
mae <- function(beta_true, beta_pred) {
  mean(abs(beta_true - beta_pred))
}

# 异常值选入率（ORA）
ora <- function(outlier_indices, sampled_indices) {
  length(intersect(outlier_indices, sampled_indices)) / length(outlier_indices)
}

#开始计算
num_simulations <- 100

# 存储所有模拟结果的列表
all_results <- list()

# 数据分布类型
dist_types <- c("normal", "t10", "t3")
# 异常值类型
outlier_types <- c("offset", "wave", "mixed")
# 异常值比例
outlier_props <- c(0.005, 0.02)









epsilon <- rnorm(n = 10000, mean = 0, sd = 1)

UNIF<-NULL;LEV<-NULL;SUBLEV<-NULL;KLEV<-NULL;TLEV<-NULL

for (dist_type in dist_types) {
  for (outlier_type in outlier_types) {
    for (outlier_prop in outlier_props) {
      unif_mse=0;unif_r2=0;unif_mae=0;unif_ora=0
      lev_mse=0;lev_r2=0;lev_mae=0;lev_ora=0
      sublev_mse = 0; sublev_r2 = 0; sublev_mae = 0; sublev_ora = 0
      klev_mse = 0;klev_r2 = 0; klev_mae = 0; klev_ora = 0
      tlev_mse = 0; tlev_r2 = 0; tlev_mae = 0; tlev_ora = 0
      
      # 存储当前组合下不同抽样方法结果的列表
      for (sim in 1:num_simulations) {
        # 生成数据
        A <- generate_data(n = 10000, p = 5, dist_type = dist_type, outlier_type = outlier_type, outlier_prop = outlier_prop)
        # 真实系数（这里假设为全1向量，实际可根据数据生成方式确定）
        beta_true <- rep(1, 5)
        # 计算真实因变量值（假设线性关系为X %*% beta_true）
        y_true <- A$X %*% beta_true+epsilon
        # 均匀抽样
        unif_result <- unif_sampling(A$X, y_true, r = 100)
               X_sampled_unif <- unif_result$X_sampled
               y_sampled_unif <- unif_result$y_sampled
               unif_pred <- predict(lm(y_sampled_unif ~ X_sampled_unif))
               unif_mse <-unif_mse+ mse(y_sampled_unif, unif_pred)
               unif_r2 <- unif_r2+r2(y_sampled_unif, unif_pred)
               unif_mae <- unif_mae+mae(beta_true, coef(lm(y_sampled_unif ~ X_sampled_unif))[-1])
               unif_ora <- unif_ora+ora(A$outlier_indices, unif_result$sampled_indices_unif)  
               # 杠杆抽样
               lev_result <- lev_sampling(A$X, y_true, r = 100)
               X_sampled_lev <- lev_result$X_sampled
               y_sampled_lev <- lev_result$y_sampled
               lev_pred <- predict(lm(y_sampled_lev ~ X_sampled_lev))
               lev_mse <-lev_mse+ mse(y_sampled_lev, lev_pred)
               lev_r2 <-lev_r2+ r2(y_sampled_lev, lev_pred)
               lev_mae <- lev_mae+mae(beta_true, coef(lm(y_sampled_lev ~ X_sampled_lev))[-1])
               lev_ora <- lev_ora+ora(A$outlier_indices, lev_result$sampled_indices)
               
               sublev_result <- sublev_sampling(A$X, y_true, outlier_prop)
               X_sampled_pure <- sublev_result$X_sampled
               y_sampled_pure <- sublev_result$y_sampled
               sublev_pred <- predict(lm(y_sampled_pure ~ X_sampled_pure))
               sublev_mse <- sublev_mse+mse(y_sampled_pure, sublev_pred)
               sublev_r2 <-sublev_r2+ r2(y_sampled_pure, sublev_pred)
               sublev_mae <- sublev_mae+mae(beta_true, coef(lm(y_sampled_pure ~ X_sampled_pure))[-1])
               sublev_ora <- sublev_ora+ora(A$outlier_indices, sublev_result$sampled_indices)
               
               klev_result <- klev_sampling(A$X, y_true, r = 100)
               X_sampled_klev <- klev_result$X_sampled
               y_sampled_klev <- klev_result$y_sampled
               klev_pred <- predict(lm(y_sampled_klev ~ X_sampled_klev))
               klev_mse <-klev_mse+ mse(y_sampled_klev, klev_pred)
               klev_r2 <- klev_r2+r2(y_sampled_klev, klev_pred)
               klev_mae <- klev_mae+mae(beta_true, coef(lm(y_sampled_klev ~ X_sampled_klev))[-1])
               klev_ora <- klev_ora+ora(A$outlier_indices, klev_result$sampled_indices)
               
               tlev_result <- tlev_sampling(A$X, y_true,r=100)
               X_sampled_robust <- tlev_result$X_sampled
               y_sampled_robust <- tlev_result$y_sampled
               tlev_pred <- predict(lm(y_sampled_robust ~ X_sampled_robust))
               tlev_mse <- tlev_mse+mse(y_sampled_robust, tlev_pred)
               tlev_r2 <- tlev_r2+r2(y_sampled_robust, tlev_pred)
               tlev_mae <- tlev_mae+mae(beta_true, coef(lm(y_sampled_robust ~ X_sampled_robust))[-1])
               tlev_ora <- tlev_ora+ora(A$outlier_indices, tlev_result$sampled_indices)
               
               
               
      }
      
      UNIF$mse <- unif_mse / num_simulations
      UNIF$r2 <- unif_r2 / num_simulations
      UNIF$mae <- unif_mae / num_simulations
      UNIF$ora <- unif_ora / num_simulations
      cat("UNIF",dist_type,outlier_type,outlier_prop, "mse:",UNIF$mse,"r2:",UNIF$r2,"mae:",UNIF$mae,"ora:",UNIF$ora,"\n")
       
      LEV$mse <- lev_mse / num_simulations
      LEV$r2 <- lev_r2 / num_simulations
      LEV$mae <- lev_mae / num_simulations
      LEV$ora <- lev_ora / num_simulations
      cat("LEV",dist_type,outlier_type,outlier_prop, "mse:",LEV$mse,"r2:",LEV$r2,"mae:",LEV$mae,"ora:",LEV$ora,"\n")
      
      SUBLEV$mse <- sublev_mse / num_simulations
      SUBLEV$r2 <- sublev_r2 / num_simulations
      SUBLEV$mae <- sublev_mae / num_simulations
      SUBLEV$ora <- sublev_ora / num_simulations
      cat("SUBLEV",dist_type,outlier_type,outlier_prop, "mse:",SUBLEV$mse,"r2:",SUBLEV$r2,"mae:",SUBLEV$mae,"ora:",SUBLEV$ora,"\n")
      
      KLEV$mse <- klev_mse / num_simulations
      KLEV$r2 <- klev_r2 / num_simulations
      KLEV$mae <- klev_mae / num_simulations
      KLEV$ora <- klev_ora / num_simulations
      cat("KLEV", dist_type, outlier_type, outlier_prop, "mse:", KLEV$mse, "r2:", KLEV$r2, "mae:", KLEV$mae, "ora:", KLEV$ora, "\n")
     
      TLEV$mse <- tlev_mse / num_simulations
      TLEV$r2 <- tlev_r2 / num_simulations
      TLEV$mae <- tlev_mae / num_simulations
      TLEV$ora <- tlev_ora / num_simulations
      cat("TLEV", dist_type, outlier_type, outlier_prop, "mse:", TLEV$mse, "r2:", TLEV$r2, "mae:", TLEV$mae, "ora:", TLEV$ora, "\n")
      
      }
  }
}