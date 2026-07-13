// Array of all slide files to load, in order.
const slideFiles = [
  'slides/01-title.html',
  'slides/02-agenda.html',
  'slides/03-divider.html',
  'slides/04-content.html',
  'slides/05-stats.html',
  'slides/06-navierstokes.html',
  'slides/07-researchquestions.html',
  'slides/08-datasets.html',
  'slides/09-models.html',
  'slides/10-training.html',
  'slides/11-results.html',
];
async function loadSlides() {
  const stage = document.getElementById('stage');
  
  // 1. Always load slides first so they are never invisible
  for (const file of slideFiles) {
    try {
      const response = await fetch(file);
      const html = await response.text();
      stage.insertAdjacentHTML('beforeend', html);
    } catch (err) {
      console.error(`Failed to load ${file}`, err);
    }
  }
  
  // 2. Fire up presentation controls right away so keys/dots work immediately
  initPresentation();

  // 3. Robustly check and apply MathJax without breaking the execution flow
  function applyMath() {
    if (window.MathJax && window.MathJax.startup && window.MathJax.typesetPromise) {
      window.MathJax.startup.promise
        .then(() => window.MathJax.typesetPromise([stage]))
        .catch(err => console.error("MathJax processing error:", err));
    } else {
      // If MathJax script hasn't arrived over the network yet, check again in 100ms
      setTimeout(applyMath, 100);
    }
  }

  applyMath();
}

function initPresentation() {
  // Stage Scaling to fit any screen
  const stage = document.getElementById('stage');
  function fitStage(){
    const w = window.innerWidth, h = window.innerHeight;
    const scale = Math.min(w / 1280, h / 720) * 0.94;
    stage.style.transform = `scale(${scale})`;
  }
  window.addEventListener('resize', fitStage);
  fitStage();

  // Slide variables
  const slides = Array.from(document.querySelectorAll('.slide'));
  const total = slides.length;
  let current = 0;

  // Generate Navigation Dots
  const dotsWrap = document.getElementById('dots');
  slides.forEach((_, i) => {
    const d = document.createElement('div');
    d.className = 'dot' + (i === 0 ? ' active' : '');
    d.addEventListener('click', () => goTo(i));
    dotsWrap.appendChild(d);
  });
  const dots = Array.from(dotsWrap.children);

  // Core Render Function
  function render(){
    slides.forEach((s, i) => {
      s.classList.toggle('is-active', i === current);
      s.classList.toggle('is-prev', i < current);
    });
    dots.forEach((d, i) => d.classList.toggle('active', i === current));
    
    // Update top progress bar width
    document.getElementById('progress-fill').style.width = `${((current + 1) / total) * 100}%`;
  }
  
  // Navigation Controls
  function goTo(i){
    current = Math.max(0, Math.min(total - 1, i));
    render();
  }
  function next(){ if (current < total - 1) goTo(current + 1); }
  function prev(){ if (current > 0) goTo(current - 1); }

  // Click Zones
  document.getElementById('zone-next').addEventListener('click', next);
  document.getElementById('zone-prev').addEventListener('click', prev);

  // Keyboard Navigation
  window.addEventListener('keydown', (e) => {
    if (['ArrowRight', 'ArrowDown', ' ', 'PageDown'].includes(e.key)) { e.preventDefault(); next(); }
    if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(e.key)) { e.preventDefault(); prev(); }
    if (e.key === 'Home') goTo(0);
    if (e.key === 'End') goTo(total - 1);
  });

  // Touch Swipe Navigation
  let touchX = null;
  window.addEventListener('touchstart', e => touchX = e.touches[0].clientX);
  window.addEventListener('touchend', e => {
    if (touchX === null) return;
    const dx = e.changedTouches[0].clientX - touchX;
    if (dx > 50) prev();
    if (dx < -50) next();
    touchX = null;
  });

  // Initial render call
  render();
}

// Boot the application
loadSlides();