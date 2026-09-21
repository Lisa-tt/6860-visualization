(() => {
  const deck = document.getElementById('deck');
  const slides = Array.from(document.querySelectorAll('.slide'));
  const mainSlides = slides.filter(slide => slide.dataset.main === 'true');
  const progress = document.getElementById('progress-fill');
  const counter = document.getElementById('slide-counter');
  const sectionIndicator = document.getElementById('section-indicator');
  const prevButton = document.getElementById('prev-button');
  const nextButton = document.getElementById('next-button');
  const notesButton = document.getElementById('notes-button');
  const notesDrawer = document.getElementById('notes-drawer');
  const notesBody = document.getElementById('notes-body');
  const notesTitle = document.getElementById('notes-title');
  const notesClose = document.getElementById('notes-close');
  const fullscreenButton = document.getElementById('fullscreen-button');
  const explorersButton = document.getElementById('explorers-button');

  const explorerOverlay = document.getElementById('explorer-overlay');
  const explorerFrame = document.getElementById('explorer-frame');
  const explorerTitle = document.getElementById('explorer-title');
  const explorerPopout = document.getElementById('explorer-popout');
  const explorerClose = document.getElementById('explorer-close');
  const explorerStage = explorerOverlay.querySelector('.explorer-stage');

  const imageOverlay = document.getElementById('image-overlay');
  const imageFull = document.getElementById('image-full');
  const imageTitle = document.getElementById('image-title');
  const imageClose = document.getElementById('image-close');

  const explorerMenu = document.getElementById('explorer-menu');
  const explorerMenuClose = document.getElementById('explorer-menu-close');

  const explorers = {
    landscape: {
      title: 'Service landscape',
      src: 'assets/interactive/01_service_landscape_explorer.html'
    },
    quality: {
      title: 'Quality geography',
      src: 'assets/interactive/02_quality_geography_explorer.html'
    },
    coverage: {
      title: 'Population-adjusted coverage',
      src: 'assets/interactive/03_population_coverage_explorer.html'
    },
    screening: {
      title: 'Multidimensional geographic screening',
      src: 'assets/interactive/04_geographic_screening_explorer.html'
    }
  };

  let current = 0;
  let touchStartX = null;

  function fitDeck() {
    const scale = Math.min(window.innerWidth / 1600, window.innerHeight / 900);
    deck.style.transform = `translate(-50%, -50%) scale(${scale})`;
  }

  function slideTitle(slide) {
    const heading = slide.querySelector('h1, h2');
    return heading ? heading.textContent.trim() : slide.dataset.section;
  }

  function updateNotes() {
    const slide = slides[current];
    const note = slide.querySelector('.speaker-note');
    notesTitle.textContent = slideTitle(slide);
    notesBody.textContent = note ? note.textContent.trim() : 'No speaker note for this slide.';
  }

  function updateCounter() {
    const mainIndex = mainSlides.indexOf(slides[current]);
    if (mainIndex >= 0) {
      counter.textContent = `${mainIndex + 1} / ${mainSlides.length}`;
      progress.style.width = `${((mainIndex + 1) / mainSlides.length) * 100}%`;
    } else {
      const appendixIndex = slides.slice(mainSlides.length).indexOf(slides[current]);
      counter.textContent = `A${appendixIndex + 1}`;
      progress.style.width = '100%';
    }
  }

  function showSlide(index, updateHash = true) {
    current = Math.max(0, Math.min(index, slides.length - 1));
    slides.forEach((slide, i) => slide.classList.toggle('is-active', i === current));
    sectionIndicator.textContent = slides[current].dataset.section || '';
    prevButton.disabled = current === 0;
    nextButton.disabled = current === slides.length - 1;
    updateCounter();
    updateNotes();
    if (updateHash) history.replaceState(null, '', `#slide-${current + 1}`);
    document.title = `${slideTitle(slides[current])} | ACECQA briefing`;
  }

  function next() { showSlide(current + 1); }
  function previous() { showSlide(current - 1); }

  function toggleNotes(force) {
    const shouldOpen = typeof force === 'boolean' ? force : !notesDrawer.classList.contains('is-open');
    notesDrawer.classList.toggle('is-open', shouldOpen);
    notesDrawer.setAttribute('aria-hidden', String(!shouldOpen));
    if (shouldOpen) updateNotes();
  }

  function openExplorer(key) {
    const explorer = explorers[key];
    if (!explorer) return;
    explorerMenu.hidden = true;
    explorerOverlay.hidden = false;
    explorerStage.classList.remove('is-loaded');
    explorerTitle.textContent = explorer.title;
    explorerFrame.src = explorer.src;
    explorerPopout.href = explorer.src;
    explorerClose.focus();
  }

  function closeExplorer() {
    explorerOverlay.hidden = true;
    explorerFrame.src = 'about:blank';
  }

  function openImage(button) {
    const image = button.querySelector('img');
    const src = button.dataset.image || image.src;
    imageFull.src = src;
    imageFull.alt = image.alt || 'Enlarged visualisation';
    imageTitle.textContent = button.title.replace(/^Enlarge\s*/i, '').replace(/^Open\s*/i, '') || 'Visualisation';
    imageOverlay.hidden = false;
    imageClose.focus();
  }

  function closeImage() {
    imageOverlay.hidden = true;
    imageFull.src = '';
  }

  prevButton.addEventListener('click', previous);
  nextButton.addEventListener('click', next);
  notesButton.addEventListener('click', () => toggleNotes());
  notesClose.addEventListener('click', () => toggleNotes(false));
  explorerClose.addEventListener('click', closeExplorer);
  imageClose.addEventListener('click', closeImage);
  explorerMenuClose.addEventListener('click', () => { explorerMenu.hidden = true; });
  explorersButton.addEventListener('click', () => { explorerMenu.hidden = false; explorerMenuClose.focus(); });

  explorerFrame.addEventListener('load', () => explorerStage.classList.add('is-loaded'));
  document.querySelectorAll('[data-explorer]').forEach(button => {
    button.addEventListener('click', () => openExplorer(button.dataset.explorer));
  });
  document.querySelectorAll('.image-button').forEach(button => {
    button.addEventListener('click', () => openImage(button));
  });

  fullscreenButton.addEventListener('click', async () => {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  });

  document.addEventListener('keydown', event => {
    const overlayOpen = !explorerOverlay.hidden || !imageOverlay.hidden || !explorerMenu.hidden;
    if (event.key === 'Escape') {
      if (!explorerOverlay.hidden) closeExplorer();
      else if (!imageOverlay.hidden) closeImage();
      else if (!explorerMenu.hidden) explorerMenu.hidden = true;
      else if (notesDrawer.classList.contains('is-open')) toggleNotes(false);
      return;
    }
    if (overlayOpen || notesDrawer.classList.contains('is-open')) return;
    if (['ArrowRight', 'ArrowDown', 'PageDown', ' '].includes(event.key)) {
      event.preventDefault();
      next();
    } else if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(event.key)) {
      event.preventDefault();
      previous();
    } else if (event.key === 'Home') {
      event.preventDefault();
      showSlide(0);
    } else if (event.key === 'End') {
      event.preventDefault();
      showSlide(mainSlides.length - 1);
    } else if (event.key.toLowerCase() === 'n') {
      toggleNotes();
    } else if (event.key.toLowerCase() === 'f') {
      fullscreenButton.click();
    }
  });

  document.addEventListener('touchstart', event => {
    touchStartX = event.changedTouches[0].screenX;
  }, { passive: true });
  document.addEventListener('touchend', event => {
    if (touchStartX === null || !explorerOverlay.hidden || !imageOverlay.hidden) return;
    const delta = event.changedTouches[0].screenX - touchStartX;
    if (Math.abs(delta) > 70) delta < 0 ? next() : previous();
    touchStartX = null;
  }, { passive: true });

  window.addEventListener('resize', fitDeck);
  window.addEventListener('hashchange', () => {
    const match = window.location.hash.match(/slide-(\d+)/);
    if (match) showSlide(Number(match[1]) - 1, false);
  });

  const initialMatch = window.location.hash.match(/slide-(\d+)/);
  fitDeck();
  showSlide(initialMatch ? Number(initialMatch[1]) - 1 : 0, false);
})();
