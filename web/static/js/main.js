/**
 * ETYMO — Main JavaScript
 * Interactive features for the Etymology-Aware Semantic Shift Analysis dashboard.
 */

document.addEventListener('DOMContentLoaded', function() {
    // ── Animate Elements on Scroll ──
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animationPlayState = 'running';
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.animate-in').forEach(el => {
        observer.observe(el);
    });

    // ── Animate Origin Bars on Load ──
    document.querySelectorAll('.origin-bar').forEach(bar => {
        const targetHeight = bar.style.height;
        bar.style.height = '0px';
        setTimeout(() => {
            bar.style.height = targetHeight;
        }, 300);
    });

    // ── Animate Drift Bars ──
    document.querySelectorAll('.drift-bar-fill').forEach(fill => {
        const targetWidth = fill.style.width;
        fill.style.width = '0%';
        setTimeout(() => {
            fill.style.width = targetWidth;
        }, 500);
    });

    // ── Global Search (redirect to explorer) ──
    const globalSearch = document.getElementById('global-search');
    if (globalSearch) {
        globalSearch.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                const word = this.value.trim().toLowerCase();
                if (word) {
                    window.location.href = `/explorer?word=${encodeURIComponent(word)}`;
                }
            }
        });

        // Autocomplete
        let timeout;
        globalSearch.addEventListener('input', function() {
            clearTimeout(timeout);
            const q = this.value.trim();
            if (q.length < 1) return;
            
            timeout = setTimeout(() => {
                fetch(`/api/search?q=${encodeURIComponent(q)}`)
                    .then(r => r.json())
                    .then(words => {
                        // Remove existing dropdown
                        const existing = document.querySelector('.search-dropdown');
                        if (existing) existing.remove();
                        
                        if (words.length === 0) return;
                        
                        const dropdown = document.createElement('div');
                        dropdown.className = 'autocomplete-list search-dropdown';
                        dropdown.style.position = 'absolute';
                        dropdown.style.top = '100%';
                        dropdown.style.left = '0';
                        dropdown.style.right = '0';
                        
                        words.forEach(w => {
                            const item = document.createElement('div');
                            item.className = 'autocomplete-item';
                            item.textContent = w;
                            item.addEventListener('click', () => {
                                window.location.href = `/explorer?word=${encodeURIComponent(w)}`;
                            });
                            dropdown.appendChild(item);
                        });
                        
                        globalSearch.parentNode.appendChild(dropdown);
                    });
            }, 200);
        });

        // Close dropdown on click outside
        document.addEventListener('click', function(e) {
            if (!e.target.closest('.header-search')) {
                const dropdown = document.querySelector('.search-dropdown');
                if (dropdown) dropdown.remove();
            }
        });
    }

    // ── Hover tooltip for drift dots ──
    document.querySelectorAll('.drift-dot').forEach(dot => {
        dot.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(2)';
            this.style.zIndex = '10';
        });
        dot.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1)';
            this.style.zIndex = '0';
        });
    });

    // ── Video poster frames ──
    document.querySelectorAll('.video-card video').forEach(video => {
        video.addEventListener('loadedmetadata', function() {
            // Seek to 2 seconds for poster frame
            this.currentTime = 2;
        });
    });

    // ── Smooth page transitions ──
    document.querySelectorAll('.nav-item, .header-nav a').forEach(link => {
        link.addEventListener('click', function(e) {
            if (this.href && !this.href.includes('#')) {
                e.preventDefault();
                document.body.style.opacity = '0.7';
                document.body.style.transition = 'opacity 0.15s';
                setTimeout(() => {
                    window.location.href = this.href;
                }, 150);
            }
        });
    });
});
