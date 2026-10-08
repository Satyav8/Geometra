/**
 * Google Apps Script: tells the chatbot to re-ingest when the FAQ sheet changes.
 *
 * SETUP (once, in the FAQ spreadsheet)
 *   1. Extensions -> Apps Script
 *   2. Paste this file over the default Code.gs, and Save
 *   3. Project Settings -> Script properties -> Add script property:
 *        INGEST_TOKEN  = the same value set on the Render service
 *      (A script property, NOT a constant in this file - anyone with edit access to the
 *       sheet can read the code, and the token triggers paid work on the service.)
 *   4. Triggers (clock icon) -> Add trigger
 *        function: onFaqChange
 *        event source: From spreadsheet
 *        event type: On change
 *   5. Run onFaqChange once manually to grant the authorisation prompt
 *
 * WHY "On change" AND NOT "On edit"
 *   onEdit does not fire for changes made by other scripts, by the Sheets API, or by a
 *   paste from another document, so a row added programmatically would never reach the
 *   bot. onChange covers those. It also fires for things that are not FAQ content at all
 *   (formatting, a comment), which is why the service hashes the content and treats an
 *   unchanged FAQ as a no-op rather than re-embedding - see rag/faq_sync.py.
 *
 * WHY THE DELAY
 *   A person adding a row types a question, then an answer, then a category: three change
 *   events seconds apart. Firing immediately each time means ingesting a half-written row
 *   and showing it to customers. This waits for the edits to stop before calling, so what
 *   gets ingested is a finished row.
 */

var ENDPOINT = 'https://geometra.onrender.com/admin/reingest';
var QUIET_PERIOD_MS = 90 * 1000;  // edits must stop for this long before a sync is sent

/** Trigger entry point. Records the change and schedules the actual call. */
function onFaqChange(e) {
  var props = PropertiesService.getScriptProperties();
  props.setProperty('pendingSince', String(Date.now()));

  // One pending timer at a time: without this, ten edits schedule ten timers and the
  // service is called ten times for one logical change.
  var triggers = ScriptApp.getProjectTriggers();
  for (var i = 0; i < triggers.length; i++) {
    if (triggers[i].getHandlerFunction() === 'flushFaqChange') {
      return;  // a flush is already scheduled; it will pick up this edit too
    }
  }
  ScriptApp.newTrigger('flushFaqChange')
    .timeBased()
    .after(QUIET_PERIOD_MS)
    .create();
}

/** Called by the timer. Calls the service only once the sheet has gone quiet. */
function flushFaqChange() {
  var props = PropertiesService.getScriptProperties();
  var pendingSince = Number(props.getProperty('pendingSince') || 0);

  // Still being edited - reschedule rather than ingesting a half-finished row.
  if (Date.now() - pendingSince < QUIET_PERIOD_MS) {
    _deleteFlushTriggers();
    ScriptApp.newTrigger('flushFaqChange').timeBased().after(QUIET_PERIOD_MS).create();
    return;
  }

  _deleteFlushTriggers();
  props.deleteProperty('pendingSince');
  syncFaqNow();
}

/** The actual call. Also safe to run by hand from the editor to test the wiring. */
function syncFaqNow() {
  var token = PropertiesService.getScriptProperties().getProperty('INGEST_TOKEN');
  if (!token) {
    Logger.log('INGEST_TOKEN script property is not set - not calling the service.');
    return;
  }

  var response = UrlFetchApp.fetch(ENDPOINT, {
    method: 'post',
    headers: {'X-Ingest-Token': token},
    muteHttpExceptions: true,  // read the status ourselves instead of throwing
  });

  var code = response.getResponseCode();
  var body = response.getContentText();
  Logger.log('reingest -> ' + code + ' ' + body);

  // 409 is the service refusing on a safety check (for example the sheet came back with
  // too few rows). That is the one case a human needs to see, because the bot is now
  // serving older content than the sheet shows and nothing else will say so.
  if (code === 409) {
    _notifyOwner('Geometra FAQ sync REFUSED', body);
  } else if (code >= 400) {
    _notifyOwner('Geometra FAQ sync failed (' + code + ')', body);
  }
}

function _deleteFlushTriggers() {
  var triggers = ScriptApp.getProjectTriggers();
  for (var i = 0; i < triggers.length; i++) {
    if (triggers[i].getHandlerFunction() === 'flushFaqChange') {
      ScriptApp.deleteTrigger(triggers[i]);
    }
  }
}

function _notifyOwner(subject, body) {
  try {
    MailApp.sendEmail(Session.getEffectiveUser().getEmail(), subject, body);
  } catch (err) {
    Logger.log('could not email the failure: ' + err);
  }
}
