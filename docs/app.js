/* Portfolio Optimization Lab — interactive demo.
   All numbers come from docs/data/demo.json, produced by scripts/make_figures.py
   running the Python library on a synthetic market. */
(function () {
  'use strict';
  var PALETTE = ['#c9a35b', '#4f86c6', '#5fb39a', '#d9775a', '#9b86d6', '#e2c88f', '#6fa8dc', '#b5c46a'];
  var NS = 'http://www.w3.org/2000/svg';
  var reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function pct(x, d) { return (x * 100).toFixed(d == null ? 1 : d) + '%'; }
  function onVisible(node, fn) {
    if (!('IntersectionObserver' in window)) { fn(); return; }
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { io.disconnect(); fn(); } });
    }, { threshold: 0.3 });
    io.observe(node);
  }
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  /* ---------- reusable animated bar group ---------- */
  function Bars(container, names, max, showVals) {
    this.max = max; this.items = [];
    var self = this;
    names.forEach(function (n, i) {
      var d = document.createElement('div'); d.className = 'bar';
      var fill = document.createElement('i'); fill.style.background = PALETTE[i % PALETTE.length];
      var lab = document.createElement('b'); lab.textContent = n;
      d.appendChild(fill); d.appendChild(lab);
      var v = null;
      if (showVals) { v = document.createElement('span'); d.appendChild(v); }
      container.appendChild(d);
      self.items.push({ fill: fill, val: v });
    });
  }
  Bars.prototype.set = function (vals) {
    var m = this.max;
    this.items.forEach(function (it, i) {
      var h = Math.max(0, Math.min(1, vals[i] / m)) * 100;
      it.fill.style.height = h + '%';
      if (it.val) { it.val.textContent = vals[i] >= 0.0005 ? pct(vals[i], 1) : ''; it.val.style.bottom = 'calc(' + h + '% + 4px)'; }
    });
  };

  fetch('data/demo.json').then(function (r) { return r.json(); }).then(init).catch(function (e) {
    console.error(e);
    document.getElementById('frontierSvg').outerHTML = '<p class="sub">Could not load demo data.</p>';
  });

  function init(D) {
    var n = D.names.length;
    function stats(w) {
      var r = 0, v = 0;
      for (var i = 0; i < n; i++) { r += w[i] * D.mu[i]; for (var j = 0; j < n; j++) v += w[i] * D.cov[i][j] * w[j]; }
      return { ret: r, vol: Math.sqrt(Math.max(v, 0)) };
    }
    var rpStats = stats(D.risk_parity.w);
    frontier(D, n, stats, rpStats);
    parity(D, n);
    residuals(D);
  }

  /* ---------- 1. Frontier ---------- */
  function frontier(D, n, stats, rpStats) {
    var svg = document.getElementById('frontierSvg');
    var W = 600, H = 400, m = { l: 52, r: 18, t: 18, b: 44 };
    // random long-only portfolios (Dirichlet(0.6) via gamma ~ -log(u)^(1/0.6) approximation: use exponential then power)
    var rnd = mulberry32(7), cloud = [];
    for (var k = 0; k < 900; k++) {
      var w = [], s = 0;
      for (var i = 0; i < n; i++) { var g = Math.pow(-Math.log(1 - rnd()), 1.6); w.push(g); s += g; }
      for (i = 0; i < n; i++) w[i] /= s;
      cloud.push(stats(w));
    }
    var xs = D.vols.concat(cloud.map(function (c) { return c.vol; }));
    var ys = D.mu.concat(cloud.map(function (c) { return c.ret; }), D.frontier.ret);
    var x0 = 0, x1 = Math.max.apply(null, xs) * 1.12;
    var y0 = Math.min(0, Math.min.apply(null, ys) - 0.01), y1 = Math.max.apply(null, ys) * 1.1;
    function X(v) { return m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r); }
    function Y(v) { return H - m.b - (v - y0) / (y1 - y0) * (H - m.t - m.b); }

    var ax = el('g', { class: 'ax' }, svg);
    var step = 0.05;
    for (var t = 0; t <= x1 + 1e-9; t += step) {
      el('line', { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b }, ax);
      el('text', { x: X(t), y: H - m.b + 16, 'text-anchor': 'middle' }, ax).textContent = pct(t, 0);
    }
    for (t = Math.ceil(y0 / step) * step; t <= y1 + 1e-9; t += step) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }, ax);
      el('text', { x: m.l - 8, y: Y(t) + 4, 'text-anchor': 'end' }, ax).textContent = pct(t, 0);
    }
    el('text', { x: (W + m.l) / 2, y: H - 6, 'text-anchor': 'middle', class: 'title' }, ax).textContent = 'Model volatility · 模型波幅';
    el('text', { x: 14, y: (H - m.b + m.t) / 2, 'text-anchor': 'middle', class: 'title', transform: 'rotate(-90 14 ' + (H - m.b + m.t) / 2 + ')' }, ax).textContent = 'Model expected return · 模型回報';

    var cg = el('g', { class: 'cloud' }, svg);
    cloud.forEach(function (c, i) {
      var dot = el('circle', { cx: X(c.vol).toFixed(1), cy: Y(c.ret).toFixed(1), r: 1.7 }, cg);
      dot.style.transitionDelay = reduce ? '0s' : (i % 60) * 18 + 'ms';
    });

    var d = D.frontier.vol.map(function (v, i) { return (i ? 'L' : 'M') + X(v).toFixed(1) + ' ' + Y(D.frontier.ret[i]).toFixed(1); }).join(' ');
    var path = el('path', { d: d, class: 'fpath' }, svg);
    var len = path.getTotalLength ? path.getTotalLength() : 1000;
    path.style.strokeDasharray = len; path.style.strokeDashoffset = len;

    var ag = el('g', { class: 'asset' }, svg);
    D.vols.forEach(function (v, i) {
      el('circle', { cx: X(v), cy: Y(D.mu[i]), r: 5.5, fill: PALETTE[i], stroke: '#0b1d33', 'stroke-width': 1.5 }, ag);
      el('text', { x: X(v) + 8, y: Y(D.mu[i]) - 6 }, ag).textContent = D.names[i];
    });

    var mk = el('g', { class: 'mk' }, svg);
    function star(cx, cy, r) {
      var p = [];
      for (var i = 0; i < 10; i++) { var a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r; p.push((cx + rr * Math.cos(a)).toFixed(1) + ',' + (cy + rr * Math.sin(a)).toFixed(1)); }
      return p.join(' ');
    }
    el('polygon', { points: star(X(D.max_sharpe.vol), Y(D.max_sharpe.ret), 10), fill: '#e2c88f', stroke: '#0b1d33', 'stroke-width': 1.2 }, mk);
    el('text', { x: X(D.max_sharpe.vol) - 12, y: Y(D.max_sharpe.ret) - 12, 'text-anchor': 'end' }, mk).textContent = 'Max Sharpe';
    el('rect', { x: X(D.min_variance.vol) - 5, y: Y(D.min_variance.ret) - 5, width: 10, height: 10, fill: '#0b1d33', stroke: '#e2c88f', 'stroke-width': 2 }, mk);
    el('text', { x: X(D.min_variance.vol) - 10, y: Y(D.min_variance.ret) + 4, 'text-anchor': 'end' }, mk).textContent = 'Min var';
    el('polygon', { points: [[0, -7], [7, 0], [0, 7], [-7, 0]].map(function (p) { return (X(rpStats.vol) + p[0]) + ',' + (Y(rpStats.ret) + p[1]); }).join(' '), fill: '#0b1d33', stroke: '#6fa8dc', 'stroke-width': 2 }, mk);
    el('line', { x1: X(rpStats.vol) + 5, y1: Y(rpStats.ret) + 5, x2: X(rpStats.vol) + 58, y2: Y(rpStats.ret) + 48, stroke: '#6fa8dc', 'stroke-width': 1 }, mk);
    el('text', { x: X(rpStats.vol) + 61, y: Y(rpStats.ret) + 56, fill: '#6fa8dc' }, mk).textContent = 'Risk parity';
    mk.style.opacity = 0; mk.style.transition = 'opacity .6s';

    var curg = el('g', { class: 'curg' }, svg);
    el('circle', { r: 9, class: 'cur-halo' }, curg);
    el('circle', { r: 7, class: 'cur' }, curg);
    curg.style.opacity = 0;

    var bars = new Bars(document.getElementById('wBars'), D.names, 1, true);
    var sVol = document.getElementById('sVol'), sRet = document.getElementById('sRet'), cap = document.getElementById('wCap');
    var slider = document.getElementById('gamma'), gV = document.getElementById('gammaV');
    var presets = document.querySelectorAll('[data-preset]');

    function show(w, vol, ret, label) {
      bars.set(w);
      sVol.textContent = pct(vol); sRet.textContent = pct(ret);
      curg.style.transform = 'translate(' + X(vol) + 'px,' + Y(ret) + 'px)';
      cap.textContent = label;
    }
    function fromSlider() {
      var i = +slider.value, R = D.risk_aversion;
      gV.textContent = 'γ = ' + R.gamma[i].toFixed(R.gamma[i] < 10 ? 1 : 0);
      presets.forEach(function (b) { b.classList.remove('on'); });
      show(R.w[i], R.vol[i], R.ret[i], 'Weights at γ = ' + R.gamma[i].toFixed(1) + ' · 權重');
    }
    slider.addEventListener('input', fromSlider);
    presets.forEach(function (b) {
      b.addEventListener('click', function () {
        presets.forEach(function (o) { o.classList.toggle('on', o === b); });
        var p = b.getAttribute('data-preset');
        if (p === 'minvar') show(D.min_variance.w, D.min_variance.vol, D.min_variance.ret, 'Minimum-variance weights · 最小方差權重');
        if (p === 'sharpe') show(D.max_sharpe.w, D.max_sharpe.vol, D.max_sharpe.ret, 'Max model Sharpe weights · 最大夏普權重');
        if (p === 'parity') show(D.risk_parity.w, rpStats.vol, rpStats.ret, 'Risk-parity weights · 風險平價權重');
      });
    });

    onVisible(svg, function () {
      cg.classList.add('show');
      setTimeout(function () {
        path.style.transition = reduce ? 'none' : 'stroke-dashoffset 2.2s cubic-bezier(.4,0,.2,1)';
        path.style.strokeDashoffset = 0;
      }, reduce ? 0 : 700);
      setTimeout(function () { mk.style.opacity = 1; curg.style.opacity = 1; fromSlider(); }, reduce ? 0 : 2700);
    });
    // place the dot without animation before it appears
    curg.style.transition = 'none';
    curg.style.transform = 'translate(' + X(D.risk_aversion.vol[+slider.value]) + 'px,' + Y(D.risk_aversion.ret[+slider.value]) + 'px)';
    requestAnimationFrame(function () { curg.style.transition = ''; });
  }

  /* ---------- 2. Risk parity ---------- */
  function parity(D, n) {
    var rc = D.risk_parity.ccd_rc, ws = D.risk_parity.ccd_w;
    var mRC = Math.max(Math.max.apply(null, rc[0]), 1 / n) * 1.3;
    var mW = Math.max(Math.max.apply(null, ws.map(function (w) { return Math.max.apply(null, w); })), 1 / n) * 1.3;
    var rcBox = document.getElementById('rcBars');
    var rcBars = new Bars(rcBox, D.names, mRC, true);
    var wBars = new Bars(document.getElementById('rpBars'), D.names, mW, true);
    var tgt = document.getElementById('rcTarget');
    tgt.style.bottom = 'calc(22px + (100% - 22px) * ' + (1 / n / mRC) + ')';
    var lbl = document.getElementById('rpLbl');
    var timer = null;
    var last = Math.min(rc.length - 1, 8);

    function frame(k) {
      rcBars.set(rc[k]); wBars.set(ws[k]);
      var dev = Math.max.apply(null, rc[k].map(function (x) { return Math.abs(x - 1 / n); }));
      lbl.textContent = (k === 0 ? 'Start: equal weights' : 'Coordinate-descent sweep ' + k) + '  ·  max deviation from 1/N: ' + pct(dev, 2);
      lbl.lang = 'en';
    }
    function play() {
      clearInterval(timer);
      var k = 0; frame(0);
      timer = setInterval(function () {
        k++;
        if (k > last) { clearInterval(timer); return; }
        frame(k);
      }, reduce ? 10 : 1100);
    }
    frame(0);
    document.getElementById('rpPlay').addEventListener('click', play);
    onVisible(rcBox, function () { setTimeout(play, 500); });
  }

  /* ---------- 3. ADMM residuals ---------- */
  function residuals(D) {
    var cv = document.getElementById('resCanvas');
    var ctx = cv.getContext('2d');
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var W = 640, H = 380;
    cv.width = W * dpr; cv.height = H * dpr; ctx.scale(dpr, dpr);
    var m = { l: 58, r: 16, t: 20, b: 40 };
    var series = D.admm.res, keys = Object.keys(series);
    var cols = ['#e2c88f', '#4f86c6', '#d9775a'];
    var iters = series[keys[0]].length;
    var all = [].concat.apply([], keys.map(function (k) { return series[k]; })).filter(function (v) { return v > 0; });
    var lmin = Math.floor(Math.log10(Math.max(Math.min.apply(null, all), 1e-13)));
    var lmax = Math.ceil(Math.log10(Math.max.apply(null, all)));
    function X(i) { return m.l + i / (iters - 1) * (W - m.l - m.r); }
    function Y(v) { return m.t + (lmax - Math.log10(Math.max(v, 1e-16))) / (lmax - lmin) * (H - m.t - m.b); }

    var leg = document.getElementById('resLegend');
    keys.forEach(function (k, i) {
      var d = document.createElement('div');
      d.innerHTML = '<i style="background:' + cols[i] + '"></i>' + k;
      leg.appendChild(d);
    });

    function axes() {
      ctx.clearRect(0, 0, W, H);
      ctx.font = '11px Inter, sans-serif'; ctx.textBaseline = 'middle';
      for (var e = lmin; e <= lmax; e += 2) {
        ctx.strokeStyle = '#1f3550'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(m.l, Y(Math.pow(10, e))); ctx.lineTo(W - m.r, Y(Math.pow(10, e))); ctx.stroke();
        ctx.fillStyle = '#93a2b5'; ctx.textAlign = 'right'; ctx.fillText('1e' + e, m.l - 8, Y(Math.pow(10, e)));
      }
      ctx.textAlign = 'center';
      for (var i = 0; i <= iters; i += 50) {
        ctx.beginPath(); ctx.moveTo(X(Math.min(i, iters - 1)), m.t); ctx.lineTo(X(Math.min(i, iters - 1)), H - m.b); ctx.stroke();
        ctx.fillText(String(i), X(Math.min(i, iters - 1)), H - m.b + 14);
      }
      ctx.fillText('Iteration · 迭代次數', (W + m.l) / 2, H - 10);
      ctx.save(); ctx.translate(14, (H - m.b + m.t) / 2); ctx.rotate(-Math.PI / 2);
      ctx.fillText('Primal residual (log) · 原始殘差', 0, 0); ctx.restore();
    }
    function draw(upto) {
      axes();
      keys.forEach(function (k, s) {
        var arr = series[k];
        ctx.strokeStyle = cols[s]; ctx.lineWidth = 2; ctx.lineJoin = 'round';
        ctx.beginPath();
        for (var i = 0; i < Math.min(upto, arr.length); i++) { var x = X(i), y = Y(arr[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
        ctx.stroke();
        if (upto > 1) {
          var j = Math.min(upto, arr.length) - 1;
          ctx.fillStyle = cols[s]; ctx.beginPath(); ctx.arc(X(j), Y(arr[j]), 3.5, 0, 7); ctx.fill();
        }
      });
    }
    var raf = null;
    function play() {
      cancelAnimationFrame(raf);
      if (reduce) { draw(iters); return; }
      var t0 = null, dur = 3200;
      function step(ts) {
        if (t0 === null) t0 = ts;
        var p = Math.min((ts - t0) / dur, 1), e = 1 - Math.pow(1 - p, 2);
        draw(Math.max(2, Math.round(e * iters)));
        if (p < 1) raf = requestAnimationFrame(step);
      }
      raf = requestAnimationFrame(step);
    }
    draw(0);
    document.getElementById('resPlay').addEventListener('click', play);
    onVisible(cv, play);
  }
})();
