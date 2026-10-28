/**
 * Help drawer
 * ===========
 * The header's Help button opens a side panel with what the current page is
 * for, the whole flow of the app and its words. The page says which help it
 * gets with <body data-help-page>; the video page adds its active tab and
 * whether the video still needs an analysis.
 */

(function () {
    const drawer = document.getElementById('helpDrawer');
    const button = document.getElementById('helpBtn');

    /**
     * Key of the "On this page" text for the page as it is right now.
     * @returns {string} Translation key under help.page
     */
    function pageKey() {
        const page = document.body.dataset.helpPage;
        if (page === 'video') {
            const app = document.getElementById('videoDetailApp');
            if (app.classList.contains('is-not-analyzed')) return 'video_not_analyzed';
            return `video_${app.dataset.tab}`;
        }
        if (page === 'person') {
            return document.getElementById('personEditor').dataset.personId ? 'person_editor' : 'person_new';
        }
        return page;
    }

    function render() {
        document.getElementById('helpPageText').textContent = t(`help.page.${pageKey()}`);
    }

    function setOpen(open) {
        drawer.hidden = !open;
        button.setAttribute('aria-expanded', String(open));
        button.classList.toggle('active', open);
        if (open) {
            render();
            drawer.focus();
        }
    }

    button.addEventListener('click', () => setOpen(drawer.hidden));
    document.getElementById('helpClose').addEventListener('click', () => {
        setOpen(false);
        button.focus();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !drawer.hidden && drawer.contains(document.activeElement)) {
            setOpen(false);
            button.focus();
        }
    });
    // Pages announce changes of context (a tab, the analysis state) with this event
    document.addEventListener('help:context', () => {
        if (!drawer.hidden) render();
    });
})();
