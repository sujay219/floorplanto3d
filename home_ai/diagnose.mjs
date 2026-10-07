import { chromium } from 'playwright';

const browser = await chromium.launch();
const page = await browser.newPage();

// Collect console logs
const logs = [];
page.on('console', msg => {
  logs.push(`[${msg.type()}] ${msg.text()}`);
});

await page.goto('http://localhost:5174/', { waitUntil: 'networkidle' });
await page.waitForTimeout(2000);

// Get diagnostics from window
const diag = await page.evaluate(() => window.__wallDiagnostics);

console.log('=== CONSOLE LOGS ===');
logs.forEach(l => console.log(l));

console.log('\n=== DIAGNOSTICS ===');
console.log(JSON.stringify(diag, null, 2));

await browser.close();
