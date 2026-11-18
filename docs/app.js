'use strict';
let metrics = [];
let currentDataset = 'ssdd';
const names = {yolo26n_seg:'YOLO26n',yolo26s_seg:'YOLO26s',yolo26m_seg:'YOLO26m',rfdetr_seg:'RF-DETR',samsar:'SAMSAR',ship_contour:'Ship Contour',swininsseg:'SwinInsSeg',solov2_gala:'SOLOv2-GALA'};
const datasetNames = {ssdd:'SSDD',hrsid:'HRSID',visible:'Visible light'};
const notes = {ssdd:'SSDD: 733 train / 183 validation / 232 test. SAR SOLOv2 is an unsuccessful diagnostic run, excluded from the ranking.',hrsid:'HRSID: 1,000 train / 250 validation / 500 test. YOLO26m has the highest autonomous mask AP50:95.',visible:'Visible light: 1,000 train / 250 validation / 500 test. All eight model variants are represented; Ship Contour is near zero.'};
function renderTable() {
  const includePrompted = document.getElementById('include-prompted').checked;
  const rows = metrics.filter(r => r.dataset === currentDataset && r.model !== 'samsar');
  rows.sort((a,b) => b.AP50_95-a.AP50_95);
  if(includePrompted) rows.push(...metrics.filter(r=>r.dataset===currentDataset && r.model==='samsar'));
  const body=document.getElementById('results-body'); body.replaceChildren();
  for (const [index,row] of rows.entries()) {
    const tr=document.createElement('tr');
    tr.className=row.model==='samsar'?'prompted':index===0?'best':'';
    const th=document.createElement('th');th.scope='row';th.textContent=names[row.model];
    if(row.model==='samsar') {const label=document.createElement('span');label.className='protocol-label';label.textContent='GT-box prompted';th.append(label);}
    tr.append(th);
    for(const key of ['AP50_95','AP50','Precision','Recall','F1','FPS']) {const td=document.createElement('td');td.textContent=Number.isFinite(row[key])?row[key].toFixed(2):'—';tr.append(td);}
    body.append(tr);
  }
  document.getElementById('dataset-note').textContent=notes[currentDataset]+(includePrompted?' SAMSAR has target boxes supplied and is not ranked against autonomous models.':'');
  document.getElementById('results-caption').textContent=datasetNames[currentDataset]+' selected final test metrics';
}
document.querySelectorAll('[data-results]').forEach(button=>button.addEventListener('click',()=>{
  currentDataset=button.dataset.results;
  document.querySelectorAll('[data-results]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
  renderTable();
}));
document.getElementById('include-prompted').addEventListener('change',renderTable);
const examples={
  hrsid:{src:'assets/hrsid-comparison.jpg',alt:'HRSID image 2010476626: input, ground truth, YOLO26m and RF-DETR predictions',caption:'HRSID image 2010476626. Both YOLO26m and RF-DETR display two instances; the instance count alone does not establish correct mask overlap. Colours distinguish instances, not prediction correctness.'},
  ssdd:{src:'assets/ssdd-comparison.jpg',alt:'SSDD image 1312896368: input, ground truth, YOLO26m and RF-DETR predictions',caption:'SSDD image 1312896368. The same scene and annotation allow differences in predicted mask extent to be inspected. Each model uses its own validation-selected threshold.'},
  visible:{src:'assets/visible-rfdetr_seg.jpg',alt:'Visible-light vessel: input, ground truth and RF-DETR predicted mask',caption:'Visible light: input, ground truth and RF-DETR prediction. The main vessel body is captured with differences around thin structures. This retained report example lacks an independently verified original test-image identifier.'}
};
document.querySelectorAll('[data-example]').forEach(button=>button.addEventListener('click',()=>{
  const example=examples[button.dataset.example];const img=document.getElementById('comparison-image');img.src=example.src;img.alt=example.alt;
  document.getElementById('comparison-caption').textContent=example.caption;
  document.querySelectorAll('[data-example]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
}));
document.querySelector('.report-preview').addEventListener('toggle',event=>{
  if(event.target.open){const frame=event.target.querySelector('iframe');if(!frame.hasAttribute('src'))frame.src=frame.dataset.src;}
});
fetch('assets/metrics.json').then(response=>{if(!response.ok)throw new Error('Metrics unavailable');return response.json();}).then(data=>{metrics=data;renderTable();}).catch(()=>{document.getElementById('dataset-note').textContent='The interactive table could not load. The chart and linked CSV remain available.';});

// Motion enhances the real experiment images; it never changes the evidence.
const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)');
const revealTargets = document.querySelectorAll('.hero-copy > *, .scope-grid > div, .section-heading, .overview > div:last-child, .summary-chart, .results-controls, .table-scroll, .finding-grid article, .curves-heading, .curve-list, .report-grid > div, .code-grid > a, .motivation-stats article, .research-purpose, .dataset-card, .annotation-story, .limitations');
let revealObserver;
if (!motionPreference.matches && 'IntersectionObserver' in window) {
  document.documentElement.classList.add('motion-ready');
  revealObserver = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        revealObserver.unobserve(entry.target);
      }
    }
  }, {threshold: 0.08, rootMargin: '0px 0px -24px 0px'});
  revealTargets.forEach((element) => {
    element.classList.add('reveal');
    // Short stagger only within small card groups, never across the whole page.
    const parent = element.parentElement;
    if (parent.matches('.scope-grid, .finding-grid, .code-grid, .hero-copy')) {
      element.style.setProperty('--reveal-delay', `${Math.min([...parent.children].indexOf(element), 4) * 65}ms`);
    }
    revealObserver.observe(element);
  });
}

for (const card of document.querySelectorAll('.hero-visual, .evidence-frame, .cover-link')) {
  card.classList.add('tilt-card');
  let animationFrame = 0;
  function resetTilt() {
    cancelAnimationFrame(animationFrame);
    card.classList.remove('is-tilting');
    card.style.setProperty('--tilt-x', '0deg');
    card.style.setProperty('--tilt-y', '0deg');
  }
  card.addEventListener('pointermove', event => {
    if (motionPreference.matches || !finePointer.matches || event.pointerType === 'touch') return;
    cancelAnimationFrame(animationFrame);
    animationFrame = requestAnimationFrame(() => {
      const bounds = card.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
      const y = Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height));
      card.style.setProperty('--tilt-x', `${(0.5 - y) * 7}deg`);
      card.style.setProperty('--tilt-y', `${(x - 0.5) * 7}deg`);
      card.style.setProperty('--pointer-x', `${x * 100}%`);
      card.style.setProperty('--pointer-y', `${y * 100}%`);
      card.classList.add('is-tilting');
    });
  });
  card.addEventListener('pointerleave', resetTilt);
  card.addEventListener('pointercancel', resetTilt);
  motionPreference.addEventListener('change', resetTilt);
}

const progress = document.createElement('div');
progress.className = 'reading-progress';
progress.setAttribute('aria-hidden', 'true');
document.body.prepend(progress);
let scrollFrame = 0;
function updateProgress() {
  const distance = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.transform = `scaleX(${distance > 0 ? Math.min(1, Math.max(0, window.scrollY / distance)) : 0})`;
  scrollFrame = 0;
}
window.addEventListener('scroll', () => {
  if (!scrollFrame) scrollFrame = requestAnimationFrame(updateProgress);
}, {passive: true});
window.addEventListener('resize', updateProgress);
document.querySelectorAll('details').forEach(detail => detail.addEventListener('toggle', updateProgress));
updateProgress();
motionPreference.addEventListener('change', () => {
  if (motionPreference.matches) {
    document.documentElement.classList.remove('motion-ready');
    if (revealObserver) revealObserver.disconnect();
  }
});
const comparisonImage = document.getElementById('comparison-image');
comparisonImage.addEventListener('load', () => {
  comparisonImage.classList.remove('is-changing');
  if (!motionPreference.matches) requestAnimationFrame(() => comparisonImage.classList.add('is-changing'));
});

// Accessible before/after viewer: original data and its annotation, not a prediction.
const annotationRange = document.getElementById('annotation-range');
if (annotationRange) {
  annotationRange.addEventListener('input', () => {
    const amount = Number(annotationRange.value);
    document.querySelector('.annotation-view').style.setProperty('--split', `${amount}%`);
    annotationRange.setAttribute('aria-valuetext', `${amount} percent annotation shown`);
  });
}
const photoDialog = document.querySelector('.photo-dialog');
if (photoDialog) {
  document.querySelectorAll('[data-photo]').forEach(button => {
    button.addEventListener('click', () => {
      const img = photoDialog.querySelector('img');
      img.src = button.dataset.photo;
      img.alt = button.querySelector('img').alt;
      document.getElementById('photo-dialog-title').textContent = button.dataset.photoTitle;
      document.getElementById('photo-dialog-caption').textContent = button.dataset.photoCaption;
      photoDialog.showModal();
    });
  });
  photoDialog.querySelector('.close-photo').addEventListener('click', () => photoDialog.close());
  photoDialog.addEventListener('click', event => {
    if (event.target !== photoDialog) return;
    const box = photoDialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) photoDialog.close();
  });
}
