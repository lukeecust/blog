document.addEventListener('DOMContentLoaded', function() {
  // 只在文章页面添加语言切换按钮
  if (document.querySelector('.post-content')) {
    // 创建语言切换按钮
    const switcherBtn = document.createElement('button');
    switcherBtn.id = 'language-toggle';
    switcherBtn.style.cssText = 'float: right; background: transparent; border: 1px solid var(--link-color); color: var(--link-color); padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9rem; cursor: pointer; margin-bottom: 1rem;';
    
    // 检查当前页面URL
    let currentPath = window.location.pathname;
    let isZhVersion = currentPath.includes('-zh');
    
    // 根据当前页面设置按钮文本
    if (isZhVersion) {
      switcherBtn.innerHTML = '切换到英文版';
    } else {
      switcherBtn.innerHTML = '切换到中文版';
    }
    
    // 设置点击事件
    switcherBtn.addEventListener('click', function() {
      // 构造对应的URL
      let newPath;
      if (isZhVersion) {
        // 当前是中文，切换到英文
        newPath = currentPath.replace(/-zh\/$/, '/');
      } else {
        // 当前是英文，切换到中文
        newPath = currentPath.replace(/\/$/, '-zh/');
      }
      window.location.href = newPath;
    });
    
    // 将按钮添加到文章头部
    const postContent = document.querySelector('.post-content');
    if (postContent) {
      postContent.parentNode.insertBefore(switcherBtn, postContent);
    }
  }
});
