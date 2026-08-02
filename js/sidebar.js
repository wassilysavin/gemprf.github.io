(function loadAssistant() {
  if (document.getElementById("gpa-assistant-css")) return;
  if (document.getElementById("ai-assistant-btn")) return;
  var v = "?v=21";  // cache
  const link = document.createElement("link");
  link.id = "gpa-assistant-css";
  link.rel = "stylesheet";
  link.href = "/css/assistant.css" + v;
  document.head.appendChild(link);
  const script = document.createElement("script");
  script.src = "/js/assistant.js" + v;
  script.defer = true;
  document.body.appendChild(script);
})();

document.addEventListener("DOMContentLoaded", () => {
  // Inject site favicon into <head> so all pages that load the sidebar get it.
  try {
    const existing = document.querySelector('link[rel="icon"]');
    if (!existing) {
      const link = document.createElement('link');
      link.rel = 'icon';
      link.type = 'image/x-icon';
      link.href = '/favicon.ico';
      document.head.appendChild(link);
    }
  } catch (e) {
    console.warn('Failed to inject favicon:', e);
  }
  // fetch and insert sidebar
  fetch("/sidebar.html")
    .then(response => response.text())
    .then(data => {
      const placeholder = document.getElementById("sidebar-placeholder");
      placeholder.innerHTML = data;

      // Prepend site logo to the sidebar header (if header exists and no logo present)
      try {
        const nav = document.querySelector('.sidebar');
        if (nav) {
          const h2 = nav.querySelector('h2');
          if (h2 && !h2.querySelector('img.sidebar-icon')) {
            // Replace h2 contents with a stacked logo and centered title
            const logoSrc = '/assets/images/icons/gem-icon-retinotopy-white.png';
            h2.innerHTML = '';
            const wrapper = document.createElement('div');
            wrapper.className = 'sidebar-logo-wrapper';

            // inner container that will be sized to the title width and centered
            const inner = document.createElement('div');
            inner.className = 'sidebar-logo-inner';

            const logoDiv = document.createElement('div');
            logoDiv.className = 'logo';
            const img = document.createElement('img');
            img.src = logoSrc;
            img.alt = 'GEM-pRF logo';
            img.className = 'sidebar-icon';
            logoDiv.appendChild(img);

            const titleDiv = document.createElement('div');
            titleDiv.className = 'logo-text';
            titleDiv.textContent = 'GEM-pRF';

            inner.appendChild(logoDiv);
            inner.appendChild(titleDiv);
            
            // Wrap the logo and title in a link to index.html
            const link = document.createElement('a');
            link.href = '/index.html';
            link.style.textDecoration = 'none';
            link.style.color = 'inherit';
            link.appendChild(inner);
            wrapper.appendChild(link);
            h2.appendChild(wrapper);

            // Size the inner container to the title width so image and title stay centered in the left-aligned wrapper.
            requestAnimationFrame(() => {
              try {
                const titleWidth = titleDiv.getBoundingClientRect().width;
                if (titleWidth && inner) {
                  inner.style.width = Math.max(0, Math.round(titleWidth)) + 'px';
                  // make image fill the inner width
                  img.style.width = '100%';
                  img.style.height = 'auto';
                }
              } catch (err) {
                // ignore measurement errors
              }
            });
          }
        }
      } catch (e) {
        console.warn('Failed to insert sidebar logo:', e);
      }

      const currentPath = window.location.pathname;

      // highlight active link + auto-expand parent collapsible
      placeholder.querySelectorAll("a").forEach(link => {
        const href = link.getAttribute("href");
        const fullHref = "/" + href.replace(/^\//, "");

        if (currentPath.endsWith(href) || currentPath.endsWith(fullHref)) {
          link.classList.add("active");

          // Expand the matching collapsible and all its ancestor collapsibles
          let p = link.closest('.collapsible');
          while (p) {
            const header = p.querySelector('.collapsible-header');
            const sublist = p.querySelector('.sublist');
            if (sublist) sublist.style.display = 'block';
            if (header) {
              if (header.textContent.includes('▸')) {
                header.textContent = header.textContent.replace('▸', '▾');
              } else if (!header.textContent.includes('▾')) {
                header.textContent = header.textContent.trim() + ' ▾';
              }
            }
            // move to the next ancestor collapsible (if any)
            const parentUL = p.parentElement;
            p = parentUL ? parentUL.closest('.collapsible') : null;
          }
        }
      });

      // initialize collapsibles
      placeholder.querySelectorAll(".collapsible").forEach(item => {
        const header = item.querySelector(".collapsible-header");
        const sublist = item.querySelector(".sublist");
        if (!header || !sublist) return;

        if (!sublist.style.display) {
          sublist.style.display = "none";
        }

        header.style.cursor = "pointer";
        header.addEventListener("click", () => {
          const isOpen = sublist.style.display === "block";
          sublist.style.display = isOpen ? "none" : "block";
          header.textContent = isOpen
            ? header.textContent.replace("▾", "▸")
            : header.textContent.replace("▸", "▾");
        });
      });
    })
    .catch(err => console.error("Sidebar load error:", err));
});
