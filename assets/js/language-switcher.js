document.addEventListener('DOMContentLoaded', function() {
  // 创建语言切换按钮
  const switcherDiv = document.createElement('div');
  switcherDiv.className = 'language-switcher';
  switcherDiv.style.cssText = 'text-align: right; margin-bottom: 1rem;';
  
  const switcherBtn = document.createElement('button');
  switcherBtn.id = 'language-toggle';
  switcherBtn.style.cssText = 'background: transparent; border: 1px solid #2a408e; color: #2a408e; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.9rem; cursor: pointer;';
  
  // 检查当前页面URL
  let currentPath = window.location.pathname;
  let isZhVersion = currentPath.includes('-zh');
  
  // 根据当前页面设置按钮文本
  if (isZhVersion) {
    switcherBtn.innerHTML = '中文 / <span style="font-weight: bold;">EN</span>';
  } else {
    switcherBtn.innerHTML = '<span style="font-weight: bold;">EN</span> / 中文';
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
  
  // 将按钮添加到文章标题区域
  const postTitle = document.querySelector('.post h1');
  if (postTitle) {
    postTitle.parentNode.insertBefore(switcherDiv, postTitle.nextSibling);
  }
});
