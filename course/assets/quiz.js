/**
 * quiz.js — Interactive quiz engine for the MIND Edge Recommender course
 * Supports: multiple-choice with feedback, open-answer reveal, score tracking, LocalStorage progress
 */

(function () {
  "use strict";

  /* ── Helpers ─────────────────────────────────────── */
  const $ = (sel, ctx = document) => ctx.querySelector(sel);
  const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

  /* ── Chapter ID from URL ─────────────────────────── */
  const chapterId = document.body.dataset.chapter || "unknown";

  /* ── LocalStorage progress ───────────────────────── */
  function getProgress() {
    try { return JSON.parse(localStorage.getItem("mind-course") || "{}"); }
    catch { return {}; }
  }
  function saveProgress(data) {
    try { localStorage.setItem("mind-course", JSON.stringify(data)); }
    catch {}
  }
  function markChapterDone(chapter, score, total) {
    const p = getProgress();
    p[chapter] = { score, total, done: score === total, ts: Date.now() };
    saveProgress(p);
    updateSidebarBadges();
  }

  /* ── Update sidebar badges ───────────────────────── */
  function updateSidebarBadges() {
    const p = getProgress();
    $$(".nav-link[data-chapter]").forEach(link => {
      const ch = link.dataset.chapter;
      const badge = link.querySelector(".badge");
      if (p[ch] && badge) {
        badge.textContent = p[ch].done ? "✓" : `${p[ch].score}/${p[ch].total}`;
        if (p[ch].done) link.classList.add("done");
      }
    });
  }

  /* ── Multiple-choice quiz ────────────────────────── */
  function initQuiz() {
    const quizSection = $(".quiz-section");
    if (!quizSection) return;

    let totalQ = 0, answeredQ = 0, score = 0;
    const scoreEl = quizSection.querySelector(".score-num");

    $$(".question-block", quizSection).forEach(block => {
      const correctIdx = parseInt(block.dataset.correct, 10);
      const opts = $$(".option", block);
      const explanation = block.querySelector(".explanation");

      if (opts.length > 0) totalQ++;

      opts.forEach((opt, idx) => {
        opt.addEventListener("click", () => {
          if (block.dataset.answered) return;
          block.dataset.answered = "1";
          answeredQ++;

          opts.forEach(o => o.classList.add("disabled"));

          if (idx === correctIdx) {
            opt.classList.add("correct");
            score++;
          } else {
            opt.classList.add("incorrect");
            opts[correctIdx]?.classList.add("reveal");
          }

          if (explanation) explanation.classList.add("show");
          if (scoreEl) scoreEl.textContent = score;

          checkQuizComplete(quizSection, score, totalQ);
        });
      });
    });

    /* ── Open-answer reveal buttons ─── */
    $$(".reveal-btn", quizSection).forEach(btn => {
      btn.addEventListener("click", () => {
        const ans = btn.previousElementSibling;
        if (ans) {
          ans.style.display = ans.style.display === "none" ? "block" : "none";
          btn.textContent = ans.style.display === "none" ? "نمایش جواب" : "پنهان کردن جواب";
        }
      });
    });
  }

  function checkQuizComplete(section, score, total) {
    if (!section) return;
    const resultEl = section.querySelector(".quiz-result");
    if (!resultEl) return;

    const answered = $$("[data-answered]", section).length;
    if (answered < total) return;

    const pct = Math.round((score / total) * 100);
    const resultScore = resultEl.querySelector(".result-score");
    const resultMsg   = resultEl.querySelector(".result-msg");

    if (resultScore) resultScore.textContent = `${pct}%`;
    if (resultMsg) {
      resultMsg.textContent =
        pct === 100 ? "عالی! همه سوال‌ها را درست جواب دادید 🎉" :
        pct >= 70   ? "خوب! با کمی تمرین بیشتر می‌توانید بهتر شوید 👍" :
                      "دوباره فصل را مرور کنید و دوباره تلاش کنید 💪";
    }
    resultEl.classList.add("show");
    markChapterDone(chapterId, score, total);
  }

  /* ── Copy button ─────────────────────────────────── */
  function initCopyButtons() {
    $$(".copy-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const pre = btn.closest(".code-wrap")?.querySelector("pre");
        if (!pre) return;
        navigator.clipboard.writeText(pre.textContent).then(() => {
          btn.textContent = "کپی شد ✓";
          btn.classList.add("copied");
          setTimeout(() => { btn.textContent = "کپی"; btn.classList.remove("copied"); }, 2000);
        });
      });
    });
  }

  /* ── Animate bars on scroll ──────────────────────── */
  function initBarCharts() {
    const bars = $$(".bar-fill[data-width]");
    if (!bars.length) return;

    bars.forEach(bar => { bar.style.width = "0"; });

    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.width = entry.target.dataset.width;
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.3 });

    bars.forEach(bar => observer.observe(bar));
  }

  /* ── Dark mode toggle ────────────────────────── */
  function initDarkMode() {
    const html = document.documentElement;
    const stored = localStorage.getItem("mind-course-theme");

    // Apply saved preference (overrides prefers-color-scheme)
    if (stored === "dark")  { html.classList.add("dark");  html.classList.remove("light"); }
    if (stored === "light") { html.classList.add("light"); html.classList.remove("dark"); }

    // Inject toggle button
    const btn = document.createElement("button");
    btn.className = "dark-toggle";
    btn.title = "تغییر حالت روشن/تاریک";
    btn.setAttribute("aria-label", "Toggle dark mode");
    const isDark = () => html.classList.contains("dark") ||
      (!html.classList.contains("light") &&
       window.matchMedia("(prefers-color-scheme: dark)").matches);
    btn.textContent = isDark() ? "☀️" : "🌙";

    btn.addEventListener("click", () => {
      if (isDark()) {
        html.classList.remove("dark");
        html.classList.add("light");
        localStorage.setItem("mind-course-theme", "light");
        btn.textContent = "🌙";
      } else {
        html.classList.remove("light");
        html.classList.add("dark");
        localStorage.setItem("mind-course-theme", "dark");
        btn.textContent = "☀️";
      }
    });

    document.body.appendChild(btn);
  }

  /* ── Init ────────────────────────────────────────── */
  document.addEventListener("DOMContentLoaded", () => {
    initDarkMode();
    initQuiz();
    initCopyButtons();
    initBarCharts();
    updateSidebarBadges();
  });
})();
