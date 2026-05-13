/* Results carousel — bulma-carousel */
document.addEventListener('DOMContentLoaded', function () {
  if (typeof bulmaCarousel !== 'undefined') {
    bulmaCarousel.attach('#results-carousel', {
      slidesToScroll: 1,
      slidesToShow: 2,
      loop: true,
      infinite: true,
      autoplay: false,
      pagination: true,
      navigation: true,
      breakpoints: [
        { changePoint: 768, slidesToShow: 1, slidesToScroll: 1 }
      ]
    });
  }

  // Smooth scroll for in-page anchors (e.g. "Video" button → #video)
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', function (e) {
      const id = this.getAttribute('href').slice(1);
      const target = document.getElementById(id);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
});
