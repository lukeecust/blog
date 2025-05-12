document.addEventListener('DOMContentLoaded', function() {
    // 创建语言切换按钮
    const switcherDiv = document.createElement('div');
    switcherDiv.className = 'language-switcher';
    
    const switcherBtn = document.createElement('button');
    switcherBtn.id = 'language-toggle';
    switcherBtn.className = 'lang-btn';
    
    // 检查当前页面URL
    let currentPath = window.location.pathname;
    let isZhVersion = currentPath.includes('-zh');
    
    // 根据当前页面设置按钮文本
    if (isZhVersion) {
      switcherBtn.innerHTML = '中文 / <span>EN</span>';
    } else {
      switcherBtn.innerHTML = '<span>EN</span> / 中文';
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
    
    switcherDiv.appendChild(switcherBtn);
    
    // 将按钮添加到文章标题区域下方
    const postTitle = document.querySelector('.post-content');
    if (postTitle) {
      postTitle.parentNode.insertBefore(switcherDiv, postTitle);
    }
  });
  