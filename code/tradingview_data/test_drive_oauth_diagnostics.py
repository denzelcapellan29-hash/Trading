"""No-credential OAuth diagnostics tests: secrets and response text never logged."""
import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from fast_drive_archive import ArchiveBlocked, DriveStore


class DiagnosticSafetyTests(unittest.TestCase):
    def create(self):
        return DriveStore('local-test-client', 'local-test-secret', 'local-test-refresh', 'folder')

    def fail_oauth(self, response):
        raw = json.dumps(response).encode()
        return HTTPError('https://oauth2.googleapis.com/token', 400, 'Bad Request', {}, io.BytesIO(raw))

    def test_reject_invalid_grant_with_identifier_only(self):
        error = self.fail_oauth({'error':'invalid_grant','error_description':'SECRET MUST NEVER LEAK'})
        with patch('fast_drive_archive.urllib.request.urlopen', side_effect=error):
            with self.assertRaises(ArchiveBlocked) as ctx: self.create()
        self.assertEqual(str(ctx.exception), 'DRIVE_OAUTH_TOKEN_EXCHANGE_FAILED:invalid_grant')
        self.assertNotIn('SECRET',str(ctx.exception))

    def test_reject_invalid_client_with_identifier_only(self):
        with patch('fast_drive_archive.urllib.request.urlopen', side_effect=self.fail_oauth({'error':'invalid_client'})):
            with self.assertRaisesRegex(ArchiveBlocked, 'FAILED:invalid_client'): self.create()

    def test_reject_undefined_response_error_without_echoing_it(self):
        with patch('fast_drive_archive.urllib.request.urlopen', side_effect=self.fail_oauth({'error':'this is not an allowed code CLIENT_SECRET=123'})):
            with self.assertRaises(ArchiveBlocked) as ctx: self.create()
        self.assertEqual(str(ctx.exception), 'DRIVE_OAUTH_TOKEN_EXCHANGE_FAILED:unclassified_response')

    def test_reject_transport_without_logging_exception(self):
        with patch('fast_drive_archive.urllib.request.urlopen', side_effect=URLError('POTENTIAL SECRET LEAK')):
            with self.assertRaises(ArchiveBlocked) as ctx: self.create()
        self.assertEqual(str(ctx.exception), 'DRIVE_OAUTH_TOKEN_EXCHANGE_FAILED:transport_error')

    def test_successful_token_exchange_does_not_display_token(self):
        with patch('fast_drive_archive.urllib.request.urlopen', return_value=io.BytesIO(b'{"access_token":"SENSITIVE"}')):
            self.assertEqual(self.create()._token,'SENSITIVE')

    def test_partial_client_details_block_before_network(self):
        with patch('fast_drive_archive.urllib.request.urlopen') as request:
            with self.assertRaisesRegex(ArchiveBlocked, 'DRIVE_OAUTH_OR_FOLDER_NOT_CONFIGURED'):
                DriveStore('', 'secret', 'token', 'folder')
            request.assert_not_called()


if __name__=='__main__': unittest.main(verbosity=2)
