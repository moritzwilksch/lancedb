// Copyright 2023 LanceDB Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import axios, { type AxiosResponse } from 'axios'

import { tableFromIPC, type Table as ArrowTable } from 'apache-arrow'

import {
  type RemoteRes,
  type RemoteRequest, Method,
  type MiddlewareContext,
  type onRemoteRequestNext,
  SimpleMiddlewareContext
} from '../middleware'

interface HttpLancedbClientMiddleware {
  onRemoteRequest(
    req: RemoteRequest,
    next: onRemoteRequestNext,
    ctx: MiddlewareContext,
  ): Promise<RemoteRes>
}

// TODO comments
async function callWithMiddlewares (
  req: RemoteRequest,
  ctx: MiddlewareContext,
  middlewares: HttpLancedbClientMiddleware[]
): Promise<RemoteRes> {
  async function call (
    i: number,
    req: RemoteRequest,
    ctx: MiddlewareContext
  ): Promise<RemoteRes> {
    // if we have reached the end of the middleware chain, make the request
    if (i > middlewares.length) {
      if (req.method !== Method.GET) {
        throw new Error('TODO unimplemented')
      }

      const res = await axios.get(
        req.uri,
        {
          headers: { ...req.headers },
          params: req.params,
          timeout: 10000
        }
      )

      return toLanceRes(res)
    }

    // call next middleware in chain
    return await middlewares[i - 1].onRemoteRequest(
      req,
      async (req, ctx) => {
        return await call(i + 1, req, ctx)
      },
      ctx
    )
  }

  return await call(1, req, ctx)
}

/**
 * Marshall the library response into a LanceDB response
 */
function toLanceRes (res: AxiosResponse): RemoteRes {
  const headers: Record<string, string> = {}
  for (const h in res.headers) {
    headers[h] = res.headers[h]
  }

  return {
    status: res.status,
    headers,
    body: async () => {
      return res.data
    }
  }
}

export class HttpLancedbClient {
  private readonly _url: string
  private readonly _apiKey: () => string
  private readonly _middlewares: HttpLancedbClientMiddleware[]
  private _middlewareCtx: MiddlewareContext

  public constructor (
    url: string,
    apiKey: string,
    private readonly _dbName?: string
  ) {
    this._url = url
    this._apiKey = () => apiKey
    this._middlewares = []
    this._middlewareCtx = new SimpleMiddlewareContext()
  }

  get uri (): string {
    return this._url
  }

  public async search (
    tableName: string,
    vector: number[],
    k: number,
    nprobes: number,
    prefilter: boolean,
    refineFactor?: number,
    columns?: string[],
    filter?: string
  ): Promise<ArrowTable<any>> {
    const response = await axios.post(
              `${this._url}/v1/table/${tableName}/query/`,
              {
                vector,
                k,
                nprobes,
                refineFactor,
                columns,
                filter,
                prefilter
              },
              {
                headers: {
                  'Content-Type': 'application/json',
                  'x-api-key': this._apiKey(),
                  ...(this._dbName !== undefined ? { 'x-lancedb-database': this._dbName } : {})
                },
                responseType: 'arraybuffer',
                timeout: 10000
              }
    ).catch((err) => {
      console.error('error: ', err)
      if (err.response === undefined) {
        throw new Error(`Network Error: ${err.message as string}`)
      }
      return err.response
    })
    if (response.status !== 200) {
      const errorData = new TextDecoder().decode(response.data)
      throw new Error(
        `Server Error, status: ${response.status as number}, ` +
        `message: ${response.statusText as string}: ${errorData}`
      )
    }

    const table = tableFromIPC(response.data)
    return table
  }

  /**
   * Sent GET request.
   */
  public async get (path: string, params?: Record<string, string | number>): Promise<RemoteRes> {
    const req = {
      uri: `${this._url}${path}`,
      method: Method.GET,
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': this._apiKey(),
        ...(this._dbName !== undefined ? { 'x-lancedb-database': this._dbName } : {})
      },
      params
    }

    try {
      const response = await callWithMiddlewares(req, this._middlewareCtx, this._middlewares)
      // TODO replace handling etc if it's not 200
      return response
    } catch (err: any) {
      console.error('error: ', err)
      if (err.response === undefined) {
        throw new Error(`Network Error: ${err.message as string}`)
      }

      return toLanceRes(err.response)
    }
  }

  /**
   * Sent POST request.
   */
  public async post (
    path: string,
    data?: any,
    params?: Record<string, string | number>,
    content?: string | undefined
  ): Promise<AxiosResponse> {
    const response = await axios.post(
        `${this._url}${path}`,
        data,
        {
          headers: {
            'Content-Type': content ?? 'application/json',
            'x-api-key': this._apiKey(),
            ...(this._dbName !== undefined ? { 'x-lancedb-database': this._dbName } : {})
          },
          params,
          timeout: 30000
        }
    ).catch((err) => {
      console.error('error: ', err)
      if (err.response === undefined) {
        throw new Error(`Network Error: ${err.message as string}`)
      }
      return err.response
    })
    if (response.status !== 200) {
      const errorData = new TextDecoder().decode(response.data)
      throw new Error(
          `Server Error, status: ${response.status as number}, ` +
          `message: ${response.statusText as string}: ${errorData}`
      )
    }
    return response
  }

  /**
   * Instrument this client with middleware
   * @param mw - The middleware that instruments the client
   * @returns - an instance of this client instrumented with the middleware
   */
  public withMiddleware (mw: HttpLancedbClientMiddleware): HttpLancedbClient {
    const wrapped = this.clone()
    wrapped._middlewares.push(mw)
    return wrapped
  }

  public withMiddlewareContext (ctx: MiddlewareContext): HttpLancedbClient {
    const wrapped = this.clone()
    wrapped._middlewareCtx = ctx
    return wrapped
  }

  /**
   * Make a clone of this client
   */
  private clone (): HttpLancedbClient {
    const clone = new HttpLancedbClient(this._url, this._apiKey(), this._dbName)
    for (const mw of this._middlewares) {
      clone._middlewares.push(mw)
    }
    return clone
  }
}
