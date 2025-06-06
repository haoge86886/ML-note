"""""
多维特征

当影响因素有多个时,要涉及到多维的线性回归
f(x) = w1*x1 + w2*x2 + w3*x3 + ....... wn*xn + b

简洁的表示: w→ = (w1,w2,w3,w4,.....,wn)
          x→ = (x1,x2,x3,x4,.....,xn)
          f(x) = w→▪x→ + b            即向量化


向量化
使用numpy的np.dot()或是torch的torch.mm()进行向量计算
当有成千上万个特征时,向量化的计算可以减少时间


多元回归的梯度下降

    {w1 = w1 ﹣α * ∂J(w1,b)/∂w1
        .
        .
        .
     wn = wn ﹣α * ∂J(wn,b)/∂wn}


    b = b ﹣α * ∂J(w,b)/∂b
得到了 w = (w1,......wn),b

"""""









