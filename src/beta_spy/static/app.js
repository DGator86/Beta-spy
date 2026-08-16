const $ = (id) => document.getElementById(id)
const history = []
let socket

function num(v, digits=2){ const n=Number(v); return Number.isFinite(n)?n.toFixed(digits):'—' }
function pct(v, digits=1){ const n=Number(v); return Number.isFinite(n)?`${(n*100).toFixed(digits)}%`:'—' }
function bps(v, digits=1){ const n=Number(v); return Number.isFinite(n)?`${n.toFixed(digits)} bp`:'—' }
function signed(v, scale=1, digits=2){ const n=Number(v); if(!Number.isFinite(n)) return '—'; const x=n*scale; return `${x>=0?'+':''}${x.toFixed(digits)}` }
function tone(el, v){ if(!el) return; el.classList.remove('pos','neg','neutral'); const n=Number(v); el.classList.add(!Number.isFinite(n)||Math.abs(n)<1e-12?'neutral':n>0?'pos':'neg') }
function gauge(id,v){ const el=$(id); const n=Math.max(-1,Math.min(1,Number(v)||0)); el.style.left=`${50+n*48}%` }
function drawSpark(){ const c=$('spark'),ctx=c.getContext('2d'),w=c.width,h=c.height; ctx.clearRect(0,0,w,h); if(history.length<2)return; const ys=history.map(x=>x.price), lo=Math.min(...ys),hi=Math.max(...ys),pad=(hi-lo||1)*.15; ctx.strokeStyle='#38d7ff';ctx.lineWidth=2;ctx.beginPath(); history.forEach((p,i)=>{const x=i/(history.length-1)*w;const y=h-((p.price-(lo-pad))/((hi+pad)-(lo-pad)))*h;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.strokeStyle='rgba(56,215,255,.14)';ctx.lineWidth=1;for(let i=1;i<4;i++){const y=i*h/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}}
function setText(id,v){ const el=$(id); if(el) el.textContent=v }

function render(state){
  setText('statusText',state.status||'UNKNOWN'); const dot=$('statusDot'); dot.style.background=state.status==='LIVE'||state.status==='DEMO'?'#45e0a8':state.status==='DEGRADED'?'#ff637b':'#ffcb66'
  setText('clock',new Date(state.timestamp||Date.now()).toLocaleString())
  setText('eventCount',`${state.stream?.events||0} stream events`); setText('lastUpdate',`last state ${state.timestamp||'—'}`)
  const s=state.snapshot; if(!s)return
  const f=s.factors||{}, syms=s.symbols||[], spy=syms.find(x=>x.symbol==='SPY')
  setText('coveragePill',`COVERAGE ${pct(f.coverage_ratio)} · ${f.symbol_count||0}/${f.expected_symbol_count||0}`)
  if(spy){ setText('spyPrice',`$${num(spy.close,2)}`); setText('spy1m',pct(spy.return_1m,2)); setText('spy5m',pct(spy.return_5m,2)); setText('spyVwap',bps(spy.vwap_distance_bps)); tone($('spy1m'),spy.return_1m);tone($('spy5m'),spy.return_5m);history.push({price:Number(spy.close),t:Date.now()});if(history.length>120)history.shift();drawSpark();setText('spyMove',pct(spy.return_5m,2));tone($('spyMove'),spy.return_5m) }
  const d=s.decision||{}; setText('decisionAction',d.action||'—');setText('decisionDirection',d.direction||'—');setText('confidence',pct(d.confidence));setText('score',num(d.score,3));$('confidenceBar').style.width=`${Math.max(0,Math.min(100,(Number(d.confidence)||0)*100))}%`;tone($('decisionDirection'),d.direction==='BULLISH'?1:d.direction==='BEARISH'?-1:0)
  const gs=$('gateStrip'); gs.innerHTML=''; Object.entries(d.gates||{}).forEach(([k,v])=>{const e=document.createElement('span');e.className=`gate ${v?'pass':'fail'}`;e.textContent=`${v?'✓':'×'} ${k.replaceAll('_',' ')}`;gs.appendChild(e)})
  setText('flowEW',signed(f.flow_ew,1,3));setText('flowW',signed(f.flow_weighted,1,3));tone($('flowEW'),f.flow_ew);tone($('flowW'),f.flow_weighted);gauge('flowEWBar',f.flow_ew);gauge('flowWBar',f.flow_weighted)
  setText('buyFlow',pct(f.pct_buy_flow));setText('participation',pct(f.participation));setText('concentration',pct(f.concentration))
  setText('trendEW',signed(f.trend_ew,1,3));setText('trendW',signed(f.trend_weighted,1,3));setText('momEW',signed(f.momentum_ew,1,3));setText('momW',signed(f.momentum_weighted,1,3));setText('volEW',num(f.volume_ew,2));setText('volW',num(f.volume_weighted,2));setText('rvEW',bps(f.volatility_ew));setText('rvW',bps(f.volatility_weighted));
  setText('aboveVwap',pct(f.pct_above_vwap));setText('emaBull',pct(f.pct_ema_bullish));setText('positive5',pct(f.pct_positive_5m));setText('breadthAccel',signed(f.breadth_acceleration,1,3))
  const fg=$('forecastGrid');fg.innerHTML='';(s.forecasts||[]).forEach(x=>{const p=Number(x.probability_up)||0;const e=document.createElement('article');e.className='forecast';e.innerHTML=`<div class="forecast-head"><b>${x.horizon_minutes}m</b><span>${x.model_ready?'MODEL LIVE':'WARMING'} · n=${x.sample_count}</span></div><div class="prob ${p>=.5?'pos':'neg'}">${(p*100).toFixed(1)}% UP</div><small>expected SPY return</small><div class="bar"><i style="width:${Math.max(0,Math.min(100,p*100))}%"></i></div><div class="forecast-foot"><span>${signed(x.expected_return_bps,1,2)} bp</span><span>conf ${(Number(x.confidence||0)*100).toFixed(0)}%</span></div>`;fg.appendChild(e)})
  const tbody=$('sectorRows');tbody.innerHTML='';(f.sectors||[]).slice().sort((a,b)=>(b.covered_weight||0)-(a.covered_weight||0)).forEach(x=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${x.sector}</td><td>${x.count}</td><td>${pct(x.covered_weight,1)}</td><td class="${(x.trend||0)>=0?'pos':'neg'}">${signed(x.trend,1,3)}</td><td class="${(x.momentum||0)>=0?'pos':'neg'}">${signed(x.momentum,1,3)}</td><td class="${(x.flow||0)>=0?'pos':'neg'}">${signed(x.flow,1,3)}</td><td>${pct(x.participation)}</td>`;tbody.appendChild(tr)})
  const lg=state.ledger; if(lg){
    setText('ledgerDay',`$${num(lg.day_realized_pnl_dollars,2)}`);tone($('ledgerDay'),lg.day_realized_pnl_dollars)
    setText('ledgerTotal',`$${num(lg.realized_pnl_dollars,2)}`);tone($('ledgerTotal'),lg.realized_pnl_dollars)
    setText('ledgerWinRate',lg.closed_count?`${pct(lg.win_rate)} of ${lg.closed_count}`:'—')
    setText('ledgerOpen',String(lg.open_count||0))
    setText('ledgerUnrealized',`$${num(lg.unrealized_pnl_dollars,2)}`);tone($('ledgerUnrealized'),lg.unrealized_pnl_dollars)
    setText('ledgerBreaker',lg.breaker_tripped?'TRIPPED — no new trades today':`ARMED (-$${num(lg.daily_loss_limit_dollars,0)})`);tone($('ledgerBreaker'),lg.breaker_tripped?-1:1)
    setText('ledgerEquity',lg.equity!=null?`$${num(lg.equity,2)}`:'—')
    setText('ledgerBudget',lg.risk_budget_dollars!=null?`$${num(lg.risk_budget_dollars,2)}`:'—')
    setText('ledgerStreak',lg.loss_streak?`${lg.loss_streak} in a row${lg.loss_streak>=3?' — budget halved':''}`:'0');tone($('ledgerStreak'),lg.loss_streak>=3?-1:0)
    const lt=$('ledgerRows');lt.innerHTML=''
    ;(lg.open_positions||[]).forEach(p=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${p.strategy}</td><td>${p.direction}</td><td>OPEN ×${p.contracts}</td><td class="${(p.unrealized_pnl_dollars||0)>=0?'pos':'neg'}">$${num(p.unrealized_pnl_dollars,2)}</td><td>—</td><td>${(p.opened_at||'').slice(11,19)}</td>`;lt.appendChild(tr)})
    ;(lg.recent_closed||[]).forEach(p=>{const tr=document.createElement('tr');tr.innerHTML=`<td>${p.strategy}</td><td>${p.direction}</td><td>CLOSED</td><td class="${(p.realized_pnl_dollars||0)>=0?'pos':'neg'}">$${num(p.realized_pnl_dollars,2)}</td><td>${p.exit_reason||'—'}</td><td>${(p.closed_at||'').slice(11,19)}</td>`;lt.appendChild(tr)})
  }
  const op=state.option_plan; if(op){$('optionEmpty').classList.add('hidden');$('optionPlan').classList.remove('hidden');setText('optionStrategy',op.strategy);setText('optionDebit',`$${num(op.debit,2)}`);setText('optionLoss',`$${num(op.max_loss_dollars,0)}`);setText('optionProfit',`$${num(op.max_profit_dollars,0)}`);const legs=$('optionLegs');legs.innerHTML='';(op.legs||[]).forEach(l=>{const e=document.createElement('div');e.className='leg';e.innerHTML=`<span>${l.side} ${l.right}${num(l.strike,0)}</span><span>${l.symbol}</span>`;legs.appendChild(e)})} else {$('optionEmpty').classList.remove('hidden');$('optionPlan').classList.add('hidden')}
}
function connect(){ socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws`);socket.onmessage=e=>{try{render(JSON.parse(e.data))}catch(err){console.error(err)}};socket.onclose=()=>setTimeout(connect,1500);socket.onerror=()=>socket.close() }
connect(); setInterval(()=>setText('clock',new Date().toLocaleString()),1000)
