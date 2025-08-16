// Professional JavaScript for Maintenance Manual Generator Website

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all features
    initSmoothScrolling();
    initAnimations();
    initStatsCounter();
    initCopyCodeButtons();
    initMobileMenu();
    initThemeToggle();
});

// Smooth scrolling for navigation links
function initSmoothScrolling() {
    const links = document.querySelectorAll('a[href^="#"]');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed nav
                
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Intersection Observer for animations
function initAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe elements that should animate
    const animatedElements = document.querySelectorAll('.feature-card, .step, .stat');
    animatedElements.forEach(el => {
        observer.observe(el);
    });
}

// Animated counter for stats
function initStatsCounter() {
    const statNumbers = document.querySelectorAll('.stat-number');
    
    const animateCounter = (element, target) => {
        const increment = target / 100;
        let current = 0;
        
        const timer = setInterval(() => {
            current += increment;
            element.textContent = Math.floor(current);
            
            if (current >= target) {
                element.textContent = target;
                clearInterval(timer);
            }
        }, 20);
    };
    
    // Create observer for stats section
    const statsObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const statNumbers = entry.target.querySelectorAll('.stat-number');
                statNumbers.forEach(stat => {
                    const target = parseInt(stat.dataset.target) || parseInt(stat.textContent);
                    stat.dataset.target = target;
                    animateCounter(stat, target);
                });
                statsObserver.unobserve(entry.target);
            }
        });
    });
    
    const statsSection = document.querySelector('.stats');
    if (statsSection) {
        statsObserver.observe(statsSection);
    }
}

// Add copy buttons to code blocks
function initCopyCodeButtons() {
    const codeBlocks = document.querySelectorAll('.code-block');
    
    codeBlocks.forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-btn';
        button.textContent = 'Copy';
        button.style.cssText = `
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: background 0.3s ease;
        `;
        
        button.addEventListener('click', () => {
            const code = block.textContent.replace('Copy', '').trim();
            navigator.clipboard.writeText(code).then(() => {
                button.textContent = 'Copied!';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        
        button.addEventListener('mouseenter', () => {
            button.style.background = 'rgba(255,255,255,0.3)';
        });
        
        button.addEventListener('mouseleave', () => {
            button.style.background = 'rgba(255,255,255,0.2)';
        });
        
        block.style.position = 'relative';
        block.appendChild(button);
    });
}

// Mobile menu toggle
function initMobileMenu() {
    const nav = document.querySelector('.nav-container');
    
    // Create mobile menu button
    const menuButton = document.createElement('button');
    menuButton.className = 'mobile-menu-btn';
    menuButton.innerHTML = '☰';
    menuButton.style.cssText = `
        display: none;
        background: none;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        position: absolute;
        top: 50%;
        right: 2rem;
        transform: translateY(-50%);
    `;
    
    // Add media query styles
    const mediaQuery = window.matchMedia('(max-width: 768px)');
    
    function handleMobileMenu(e) {
        if (e.matches) {
            menuButton.style.display = 'block';
            nav.style.flexDirection = 'column';
        } else {
            menuButton.style.display = 'none';
            nav.style.flexDirection = 'row';
            nav.style.display = 'flex';
        }
    }
    
    mediaQuery.addListener(handleMobileMenu);
    handleMobileMenu(mediaQuery);
    
    menuButton.addEventListener('click', () => {
        nav.style.display = nav.style.display === 'none' ? 'flex' : 'none';
    });
    
    document.querySelector('.nav').appendChild(menuButton);
}

// Theme toggle (optional feature)
function initThemeToggle() {
    // This could be expanded to include dark mode toggle
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
    
    // Future implementation for dark mode
    if (prefersDarkScheme.matches) {
        // Apply dark theme styles if needed
        console.log('User prefers dark mode');
    }
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Scroll to top functionality
function createScrollToTop() {
    const button = document.createElement('button');
    button.className = 'scroll-to-top';
    button.innerHTML = '↑';
    button.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: var(--secondary-color);
        color: white;
        border: none;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        font-size: 1.2rem;
        cursor: pointer;
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    `;
    
    button.addEventListener('click', () => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    
    window.addEventListener('scroll', debounce(() => {
        if (window.pageYOffset > 300) {
            button.style.opacity = '1';
            button.style.visibility = 'visible';
        } else {
            button.style.opacity = '0';
            button.style.visibility = 'hidden';
        }
    }, 100));
    
    document.body.appendChild(button);
}

// Initialize scroll to top
createScrollToTop();

// Add loading animation
window.addEventListener('load', () => {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.3s ease';
    
    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
});

// Error handling for images
document.querySelectorAll('img').forEach(img => {
    img.addEventListener('error', function() {
        this.style.display = 'none';
        console.warn('Image failed to load:', this.src);
    });
});

// Progressive enhancement for older browsers
if (!window.IntersectionObserver) {
    // Fallback for browsers without Intersection Observer
    const fallbackElements = document.querySelectorAll('.feature-card, .step, .stat');
    fallbackElements.forEach(el => {
        el.classList.add('fade-in');
    });
}
