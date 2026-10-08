export function plot(canvas,row,heat=false){
 const dpr=Math.min(devicePixelRatio,2),w=canvas.clientWidth||400,h=175;canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);
 const box={x:46,y:13,w:w-64,h:h-45};c.fillStyle='#fff';c.fillRect(0,0,w,h);c.font='9px Segoe UI, sans-serif';c.strokeStyle='#e9edf3';c.fillStyle='#8a97a7';
 if(!heat){const xs=row.frequency_khz,ys=row.psd_dbm_hz;let low=Math.floor(Math.min(...ys)/10)*10,high=Math.ceil(Math.max(...ys)/10)*10;if(high===low)high=low+10;
  for(let i=0;i<=4;i++){const y=box.y+i*box.h/4;c.beginPath();c.moveTo(box.x,y);c.lineTo(box.x+box.w,y);c.stroke();c.fillText((high-i*(high-low)/4).toFixed(0),6,y+3);}
  c.beginPath();ys.forEach((v,i)=>{const x=box.x+i/(ys.length-1)*box.w,y=box.y+(high-v)/(high-low)*box.h;i?c.lineTo(x,y):c.moveTo(x,y);});c.strokeStyle='#377bef';c.lineWidth=1.25;c.stroke();
  for(let i=0;i<=4;i++)c.fillText((xs[0]+i*(xs.at(-1)-xs[0])/4).toFixed(0),box.x+i*box.w/4-10,h-17);
  c.fillText('kHz',w-26,h-17);c.fillText('dBm/Hz',4,8);
 }else{const z=row.spectrogram_dbm_hz,ny=z.length,nx=z[0].length,[low,high]=row.color_range;
  const bitmap=document.createElement('canvas');bitmap.width=nx;bitmap.height=ny;const bc=bitmap.getContext('2d'),im=bc.createImageData(nx,ny);
  const palette=[[55,17,94],[61,71,137],[33,144,141],[92,200,99],[253,231,37]];
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){let v=Math.max(0,Math.min(.9999,(z[ny-1-y][x]-low)/Math.max(high-low,1)));const k=Math.floor(v*4),f=v*4-k,idx=(y*nx+x)*4;for(let j=0;j<3;j++)im.data[idx+j]=palette[k][j]*(1-f)+palette[k+1][j]*f;im.data[idx+3]=255;}
  bc.putImageData(im,0,0);c.imageSmoothingEnabled=false;c.drawImage(bitmap,box.x,box.y,box.w,box.h);
  const freq=row.stft_frequency_khz;for(let i=0;i<=4;i++){c.fillText((freq.at(-1)-i*(freq.at(-1)-freq[0])/4).toFixed(0),4,box.y+i*box.h/4+3);c.fillText((row.time_ms[0]+i*(row.time_ms.at(-1)-row.time_ms[0])/4).toFixed(1),box.x+i*box.w/4-8,h-17);}
  c.fillText('ms',w-22,h-17);c.fillText('kHz',6,8);c.fillText(`${low.toFixed(0)} — ${high.toFixed(0)} dBm/Hz`,box.x,8);
 }
}
